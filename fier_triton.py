from __future__ import annotations

"""Triton kernels for FIER's group-wise 1-bit key retrieval.

The layout follows the FIER authors' public implementation: keys are grouped
along the token axis, quantized with per-channel min/max metadata, and packed
32-to-1 before a fused dequantization/GEMV kernel.  This implementation adds
partial-group handling and GQA head mapping needed by this evaluator.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _group_minmax_kernel(
    keys,
    mins,
    maxs,
    tokens,
    head_dim: tl.constexpr,
    num_groups,
    group_size: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    d = row % head_dim
    group = (row // head_dim) % num_groups
    kvh = row // (head_dim * num_groups)
    offsets = tl.arange(0, BLOCK_GROUP)
    token = group * group_size + offsets
    mask = (offsets < group_size) & (token < tokens)
    values = tl.load(
        keys + (kvh * tokens + token) * head_dim + d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    lo = tl.min(tl.where(mask, values, float("inf")), axis=0)
    hi = tl.max(tl.where(mask, values, -float("inf")), axis=0)
    metadata_offset = (kvh * num_groups + group) * head_dim + d
    tl.store(mins + metadata_offset, lo)
    tl.store(maxs + metadata_offset, hi)


@triton.jit
def _pack_kernel(
    keys,
    mins,
    maxs,
    packed,
    tokens,
    head_dim: tl.constexpr,
    num_groups,
    words,
    group_size: tl.constexpr,
):
    row = tl.program_id(0)
    word = row % words
    d = (row // words) % head_dim
    kvh = row // (words * head_dim)
    lane = tl.arange(0, 32)
    token = word * 32 + lane
    group = token // group_size
    metadata_offset = (kvh * num_groups + group) * head_dim + d
    lo = tl.load(mins + metadata_offset, mask=token < tokens, other=0.0)
    hi = tl.load(maxs + metadata_offset, mask=token < tokens, other=0.0)
    value = tl.load(
        keys + (kvh * tokens + token) * head_dim + d,
        mask=token < tokens,
        other=0.0,
    ).to(tl.float32)
    midpoint = (lo + hi) * 0.5
    bit = (value > midpoint) & (token < tokens)
    # Accumulate in int64 so lane 31 does not overflow during the reduction;
    # storing to int32 preserves the desired bit pattern.
    word_value = tl.sum(bit.to(tl.int64) << lane, axis=0)
    tl.store(packed + (kvh * head_dim + d) * words + word, word_value)


def _validate_keys(keys: torch.Tensor, group_size: int) -> None:
    if keys.ndim != 3:
        raise ValueError("keys must be [KVH, T, D]")
    if not keys.is_cuda:
        raise RuntimeError("FIER Triton backend requires CUDA tensors")
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    if keys.shape[1] <= 0 or keys.shape[2] <= 0:
        raise ValueError("keys must have non-empty token and head dimensions")
    if group_size <= 0 or group_size > 1024:
        raise ValueError("group_size must be in [1, 1024]")


def pack_keys(
    keys: torch.Tensor, *, group_size: int, implementation: str = "author"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack authors-style FIER keys as [KVH, token_word, D]."""

    keys = keys.contiguous()
    _validate_keys(keys, int(group_size))
    if implementation != "author":
        raise ValueError("only the FIER authors implementation is public")
    if implementation == "author" and group_size != 32:
        raise ValueError("authors-style FIER requires group_size=32")
    kv_heads, tokens, head_dim = (int(v) for v in keys.shape)
    num_groups = math.ceil(tokens / group_size)
    words = math.ceil(tokens / 32)
    metadata_shape = (kv_heads, num_groups, head_dim)
    # Match the authors’ cache layout: min/max metadata uses the source key
    # dtype. Group extrema are source values, so this does not add quantization error.
    mins = torch.empty(metadata_shape, device=keys.device, dtype=keys.dtype)
    maxs = torch.empty_like(mins)
    packed = torch.empty(
        (kv_heads, head_dim, words), device=keys.device, dtype=torch.int32
    )
    block_group = triton.next_power_of_2(group_size)
    with torch.cuda.device(keys.device):
        _group_minmax_kernel[(kv_heads * num_groups * head_dim,)](
            keys,
            mins,
            maxs,
            tokens=tokens,
            head_dim=head_dim,
            num_groups=num_groups,
            group_size=group_size,
            BLOCK_GROUP=block_group,
            num_warps=min(8, max(1, block_group // 32)),
        )
        _pack_kernel[(kv_heads * head_dim * words,)](
            keys,
            mins,
            maxs,
            packed,
            tokens=tokens,
            head_dim=head_dim,
            num_groups=num_groups,
            words=words,
            group_size=group_size,
            num_warps=1,
        )
    packed = packed.transpose(1, 2).contiguous()
    maxs.sub_(mins)  # Author notation: zero=min, scale=max-min.
    return packed, mins, maxs


def score_packed_batched(
    queries: torch.Tensor,
    packed: torch.Tensor,
    zero: torch.Tensor,
    scale: torch.Tensor,
    head_to_kv: torch.Tensor,
    *,
    tokens: int,
    group_size: int,
    implementation: str = "author",
) -> torch.Tensor:
    """Authors-style fused 1-bit dequantization/GEMV for [B,QH,D]."""
    if implementation != "author":
        raise ValueError("only the FIER authors implementation is public")
    if group_size != 32:
        raise ValueError("FIER-author requires group_size=32")
    if queries.ndim != 3 or packed.ndim != 3:
        raise ValueError("queries and packed keys must be rank 3")
    if not queries.is_cuda or queries.device != packed.device:
        raise RuntimeError("all FIER tensors must share a CUDA device")
    _, q_heads, head_dim = (int(v) for v in queries.shape)
    if int(packed.shape[2]) != head_dim:
        raise ValueError("authors-style packed key must be [KVH, words, head_dim]")
    if zero.shape != scale.shape or zero.ndim != 3:
        raise ValueError("authors-style zero/scale tensors must have matching shapes")
    expected_groups = math.ceil(tokens / group_size)
    if tokens <= 0 or tokens > int(packed.shape[1]) * 32:
        raise ValueError("tokens is inconsistent with authors-style packed storage")
    if int(zero.shape[0]) != int(packed.shape[0]) or int(zero.shape[1]) < expected_groups:
        raise ValueError("authors-style metadata capacity is inconsistent with tokens")
    if int(zero.shape[2]) != head_dim or head_to_kv.numel() != q_heads:
        raise ValueError("authors-style head dimensions or mapping are inconsistent")
    from popkv_cuda import score_fier_author
    return score_fier_author(
        queries.contiguous(), packed, zero, scale,
        head_to_kv=head_to_kv.to(device=queries.device, dtype=torch.int64).contiguous(),
        tokens=tokens,
    )


def score_tensors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    group_size: int,
    implementation: str = "author",
) -> torch.Tensor:
    packed, mins, maxs = pack_keys(keys, group_size=group_size, implementation=implementation)
    return score_packed_batched(
        queries,
        packed,
        mins,
        maxs,
        head_to_kv,
        tokens=int(keys.shape[1]),
        group_size=group_size,
        implementation=implementation,
    )
