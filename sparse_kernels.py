from __future__ import annotations

"""Optimized Triton kernels shared by Quest, FIER, and PopKV decoding."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_score_kernel(
    queries,
    page_min,
    page_max,
    head_to_kv,
    scores,
    pages,
    q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    min_stride_h,
    min_stride_p,
    max_stride_h,
    max_stride_p,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_block = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = tl.load(head_to_kv + qh)
    page = page_block * BLOCK_P + tl.arange(0, BLOCK_P)
    dim = tl.arange(0, BLOCK_D)
    mask = (page[:, None] < pages) & (dim[None, :] < head_dim)
    query = tl.load(queries + qh * head_dim + dim, mask=dim < head_dim, other=0.0).to(tl.float32)
    lo = tl.load(
        page_min + kvh * min_stride_h + page[:, None] * min_stride_p + dim[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    hi = tl.load(
        page_max + kvh * max_stride_h + page[:, None] * max_stride_p + dim[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bound = tl.maximum(query[None, :] * lo, query[None, :] * hi)
    result = tl.sum(bound, axis=1)
    tl.store(scores + qh * pages + page, result, mask=page < pages)


def quest_scores(
    queries: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
    head_to_kv: torch.Tensor,
    *,
    pages: int,
) -> torch.Tensor:
    """Fused Quest upper-bound scoring without query-head metadata copies."""

    if queries.ndim != 2 or page_min.ndim != 3 or page_max.shape != page_min.shape:
        raise ValueError("expected queries [QH,D] and matching page bounds [KVH,P,D]")
    if not queries.is_cuda or queries.device != page_min.device or page_max.device != queries.device:
        raise RuntimeError("Quest Triton tensors must share a CUDA device")
    q_heads, head_dim = (int(x) for x in queries.shape)
    if pages <= 0 or pages > int(page_min.shape[1]) or int(page_min.shape[2]) != head_dim:
        raise ValueError("invalid Quest page count or head dimension")
    mapping = head_to_kv.to(device=queries.device, dtype=torch.int64).contiguous()
    scores = torch.empty((q_heads, pages), device=queries.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    block_p = 16
    with torch.cuda.device(queries.device):
        _quest_score_kernel[(triton.cdiv(pages, block_p), q_heads)](
            queries.contiguous(), page_min, page_max, mapping, scores, pages,
            q_heads=q_heads, head_dim=head_dim,
            min_stride_h=page_min.stride(0), min_stride_p=page_min.stride(1),
            max_stride_h=page_max.stride(0), max_stride_p=page_max.stride(1),
            BLOCK_P=block_p, BLOCK_D=block_d, num_warps=4,
        )
    return scores


@triton.jit
def _indexed_attention_kernel(
    queries,
    prefix_keys,
    prefix_values,
    current_keys,
    current_values,
    indices,
    valid_mask,
    head_to_kv,
    output,
    prefix_tokens,
    selected_tokens,
    stride_kh,
    stride_kt,
    stride_vh,
    stride_vt,
    q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SELECTED: tl.constexpr,
):
    qh = tl.program_id(0)
    kvh = tl.load(head_to_kv + qh)
    dim = tl.arange(0, BLOCK_D)
    dim_mask = dim < head_dim
    query = tl.load(queries + qh * head_dim + dim, mask=dim_mask, other=0.0).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start in range(0, MAX_SELECTED, BLOCK_T):
        offset = start + tl.arange(0, BLOCK_T)
        token_mask = offset < selected_tokens
        token = tl.load(indices + qh * selected_tokens + offset, mask=token_mask, other=0)
        if HAS_VALID:
            token_mask = token_mask & tl.load(
                valid_mask + qh * selected_tokens + offset, mask=token_mask, other=0
            ).to(tl.int1)
        is_current = token == prefix_tokens
        safe_token = tl.minimum(token, prefix_tokens - 1)
        tile_mask = token_mask[:, None] & dim_mask[None, :]

        cached_key = tl.load(
            prefix_keys + kvh * stride_kh + safe_token[:, None] * stride_kt + dim[None, :],
            mask=tile_mask,
            other=0.0,
        ).to(tl.float32)
        current_key = tl.load(
            current_keys + kvh * head_dim + dim,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        key = tl.where(is_current[:, None], current_key[None, :], cached_key)
        logits = tl.sum(key * query[None, :], axis=1) * scale
        logits = tl.where(token_mask, logits, -float("inf"))

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(logits - new_max)
        probabilities = tl.where(token_mask, probabilities, 0.0)

        cached_value = tl.load(
            prefix_values + kvh * stride_vh + safe_token[:, None] * stride_vt + dim[None, :],
            mask=tile_mask,
            other=0.0,
        ).to(tl.float32)
        current_value = tl.load(
            current_values + kvh * head_dim + dim,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.where(is_current[:, None], current_value[None, :], cached_value)
        accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * value, axis=0)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max

    tl.store(output + qh * head_dim + dim, accumulator / running_sum, mask=dim_mask)


def indexed_attention(
    queries: torch.Tensor,
    prefix_keys: torch.Tensor,
    prefix_values: torch.Tensor,
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    indices: torch.Tensor,
    head_to_kv: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused indexed K/V loading, softmax, and q_len=1 attention."""

    if queries.ndim != 2 or prefix_keys.ndim != 3 or prefix_values.shape != prefix_keys.shape:
        raise ValueError("invalid indexed-attention tensor ranks or K/V shapes")
    if indices.ndim != 2 or int(indices.shape[0]) != int(queries.shape[0]):
        raise ValueError("indices must be [query heads, selected tokens]")
    if valid_mask is not None and valid_mask.shape != indices.shape:
        raise ValueError("valid mask must match indices")
    if not queries.is_cuda or any(x.device != queries.device for x in (
        prefix_keys, prefix_values, current_keys, current_values, indices, head_to_kv
    )):
        raise RuntimeError("indexed-attention tensors must share a CUDA device")
    q_heads, head_dim = (int(x) for x in queries.shape)
    prefix_tokens = int(prefix_keys.shape[1])
    selected_tokens = int(indices.shape[1])
    if prefix_tokens <= 0 or selected_tokens <= 0:
        raise ValueError("indexed attention requires non-empty prefix and selection")
    max_selected = triton.next_power_of_2(selected_tokens)
    output = torch.empty_like(queries)
    dummy_valid = indices if valid_mask is None else valid_mask.contiguous()
    with torch.cuda.device(queries.device):
        _indexed_attention_kernel[(q_heads,)](
            queries.contiguous(), prefix_keys, prefix_values,
            current_keys.contiguous(), current_values.contiguous(), indices.contiguous(),
            dummy_valid, head_to_kv.to(dtype=torch.int64).contiguous(), output,
            prefix_tokens, selected_tokens,
            stride_kh=prefix_keys.stride(0), stride_kt=prefix_keys.stride(1),
            stride_vh=prefix_values.stride(0), stride_vt=prefix_values.stride(1),
            q_heads=q_heads, head_dim=head_dim, scale=1.0 / math.sqrt(head_dim),
            HAS_VALID=valid_mask is not None, BLOCK_T=16,
            BLOCK_D=triton.next_power_of_2(head_dim), MAX_SELECTED=max_selected,
            num_warps=4,
        )
    return output


# ---- Quest exact page selection ---------------------------------------------

@triton.jit
def _quest_expand_kernel(
    selected_pages,
    indices,
    valid,
    num_tokens,
    page_budget: tl.constexpr,
    page_size: tl.constexpr,
    output_tokens: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    qh = tl.program_id(0)
    page_slot = tl.arange(0, BLOCK_P)
    pages = tl.load(
        selected_pages + qh * page_budget + page_slot,
        mask=page_slot < page_budget,
        other=-1,
    )
    current = num_tokens - 1
    current_page = current // page_size
    has_current = tl.max(tl.where(page_slot < page_budget, pages == current_page, 0), axis=0)

    output_slot = tl.arange(0, BLOCK_OUT)
    output_mask = output_slot < output_tokens
    source_page_slot = output_slot // page_size
    source_page = tl.load(
        selected_pages + qh * page_budget + source_page_slot,
        mask=output_mask,
        other=0,
    )
    token = source_page * page_size + output_slot % page_size
    token_valid = output_mask & (token < num_tokens)
    is_last = output_slot == output_tokens - 1
    token = tl.where(is_last & (has_current == 0), current, token)
    token_valid = tl.where(is_last & (has_current == 0), 1, token_valid)
    tl.store(indices + qh * output_tokens + output_slot, token, mask=output_mask)
    tl.store(valid + qh * output_tokens + output_slot, token_valid, mask=output_mask)


def quest_indices(
    scores: torch.Tensor, *, num_tokens: int, page_size: int, budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native exact page top-k plus fused page expansion/current reservation."""

    if scores.ndim != 2 or not scores.is_cuda:
        raise ValueError("Quest scores must be CUDA [query heads, pages]")
    page_budget = min(int(scores.shape[-1]), max(1, math.ceil(budget / page_size)))
    selected_pages = torch.topk(scores, k=page_budget, dim=-1, sorted=False).indices
    output_tokens = page_budget * page_size
    indices = torch.empty(
        (int(scores.shape[0]), output_tokens), device=scores.device, dtype=torch.long,
    )
    valid = torch.empty_like(indices, dtype=torch.bool)
    with torch.cuda.device(scores.device):
        _quest_expand_kernel[(int(scores.shape[0]),)](
            selected_pages, indices, valid, num_tokens,
            page_budget=page_budget, page_size=page_size, output_tokens=output_tokens,
            BLOCK_P=triton.next_power_of_2(page_budget),
            BLOCK_OUT=triton.next_power_of_2(output_tokens), num_warps=8,
        )
    return indices, valid


