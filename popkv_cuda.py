from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("Pop-KV CUDA scoring requires an NVIDIA GPU")
    return load(
        name="popkv_ext",
        sources=[str(_ROOT / "csrc" / "popkv_ext.cpp"), str(_ROOT / "csrc" / "popkv_ext.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def _validate_dim_group_size(dim_group_size: int) -> int:
    value = int(dim_group_size)
    if value not in (16, 32, 64):
        raise ValueError("dim_group_size must be one of 16, 32, 64")
    return value


def pack_keys(
    keys: torch.Tensor, *, token_capacity: int | None = None, dim_group_size: int = 32,
):
    """Pack fixed uint32 K bitplanes plus G_dim low/delta representatives."""
    if keys.ndim != 3 or not keys.is_cuda or not keys.is_contiguous():
        raise ValueError("keys must be contiguous CUDA [KV heads, tokens, head dim]")
    capacity = int(keys.shape[1]) if token_capacity is None else int(token_capacity)
    dim_group_size = _validate_dim_group_size(dim_group_size)
    if dim_group_size == 32:
        return tuple(_extension().pack(keys, capacity))
    return tuple(_extension().pack_dim(keys, capacity, dim_group_size))


def pack_keys_into(
    keys: torch.Tensor, packed, *, token_offset: int, dim_group_size: int = 32,
) -> None:
    if keys.ndim != 3 or not keys.is_cuda or not keys.is_contiguous():
        raise ValueError("keys must be contiguous CUDA [KV heads, tokens, head dim]")
    dim_group_size = _validate_dim_group_size(dim_group_size)
    if dim_group_size == 32:
        _extension().pack_into(keys, *packed, int(token_offset))
    else:
        _extension().pack_into_dim(keys, *packed, int(token_offset), dim_group_size)


def allocate_query_encode_outputs(queries: torch.Tensor, *, dim_group_size: int = 32):
    """Allocate benchmark/debug outputs matching the fused scorer's query state."""
    if queries.ndim != 3 or not queries.is_cuda or not queries.is_contiguous():
        raise ValueError("queries must be contiguous CUDA [batch, query heads, head dim]")
    words = (int(queries.shape[2]) + 31) // 32
    dim_group_size = _validate_dim_group_size(dim_group_size)
    groups = (int(queries.shape[2]) + dim_group_size - 1) // dim_group_size
    bit_shape = (int(queries.shape[0]), int(queries.shape[1]), words)
    rep_shape = (int(queries.shape[0]), int(queries.shape[1]), groups)
    sign = torch.empty(bit_shape, device=queries.device, dtype=torch.int32)
    magnitude = torch.empty_like(sign)
    low = torch.empty(rep_shape, device=queries.device, dtype=torch.float32)
    delta = torch.empty_like(low)
    return sign, magnitude, low, delta


def query_encode(queries: torch.Tensor, *, out=None, dim_group_size: int = 32):
    """Run the benchmark-only dimension-group query encoder; production is unchanged."""
    if queries.ndim != 3 or not queries.is_cuda or not queries.is_contiguous():
        raise ValueError("queries must be contiguous CUDA [batch, query heads, head dim]")
    dim_group_size = _validate_dim_group_size(dim_group_size)
    outputs = (allocate_query_encode_outputs(queries, dim_group_size=dim_group_size)
               if out is None else tuple(out))
    if len(outputs) != 4:
        raise ValueError("out must contain q_sign, q_mag, q_low, q_delta")
    _extension().query_encode_out(queries, *outputs, dim_group_size)
    return outputs


def score(
    queries: torch.Tensor, packed, *, head_to_kv: torch.Tensor, tokens: int,
    dim_group_size: int = 32,
) -> torch.Tensor:
    """Return Pop-KV approximate QK scores for every query head and token."""
    if queries.ndim != 3 or not queries.is_cuda or not queries.is_contiguous():
        raise ValueError("queries must be contiguous CUDA [batch, query heads, head dim]")
    dim_group_size = _validate_dim_group_size(dim_group_size)
    if dim_group_size == 32:
        return _extension().score(queries, *packed, head_to_kv.contiguous(), int(tokens))
    return _extension().score_dim(
        queries, *packed, head_to_kv.contiguous(), int(tokens), dim_group_size,
    )
