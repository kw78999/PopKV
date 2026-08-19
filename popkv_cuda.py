from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
MAX_TOKEN_GROUP_SIZE = 4096
SPECIALIZED_TOKEN_GROUP_SIZES = (8, 16, 32, 64, 128)
SUPPORTED_DIM_KERNELS = ("g16", "g32", "g64-u32", "g64-u64")


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


def _validate_token_group_size(token_group_size: int) -> int:
    value = int(token_group_size)
    if not 1 <= value <= MAX_TOKEN_GROUP_SIZE:
        raise ValueError(f"token_group_size must be between 1 and {MAX_TOKEN_GROUP_SIZE}")
    return value


def _validate_dim_group_size(dim_group_size: int) -> int:
    value = int(dim_group_size)
    if value not in (16, 32, 64):
        raise ValueError("dim_group_size must be one of 16, 32, 64")
    return value


def _validate_dim_kernel(dim_group_size: int, dim_kernel: str | None) -> str:
    default = {16: "g16", 32: "g32", 64: "g64-u32"}[dim_group_size]
    value = default if dim_kernel is None or dim_kernel == "auto" else str(dim_kernel)
    if value not in SUPPORTED_DIM_KERNELS:
        raise ValueError(f"dim_kernel must be one of {SUPPORTED_DIM_KERNELS}")
    expected = 64 if value.startswith("g64-") else int(value[1:])
    if expected != dim_group_size:
        raise ValueError(f"dim_kernel={value} requires dim_group_size={expected}")
    return value


def pack_keys(
    keys: torch.Tensor, *, token_capacity: int | None = None,
    token_group_size: int = 32, dim_group_size: int = 32,
):
    """Pack fixed uint32 K bitplanes plus G_dim low/delta representatives."""
    if keys.ndim != 3 or not keys.is_cuda or not keys.is_contiguous():
        raise ValueError("keys must be contiguous CUDA [KV heads, tokens, head dim]")
    capacity = int(keys.shape[1]) if token_capacity is None else int(token_capacity)
    token_group_size = _validate_token_group_size(token_group_size)
    dim_group_size = _validate_dim_group_size(dim_group_size)
    if dim_group_size != 32 and token_group_size not in SPECIALIZED_TOKEN_GROUP_SIZES:
        raise ValueError("non-specialized token group sizes require dim_group_size=32")
    if dim_group_size == 32:
        return tuple(_extension().pack(keys, capacity, token_group_size))
    return tuple(_extension().pack_dim(
        keys, capacity, dim_group_size, token_group_size,
    ))


def pack_keys_into(
    keys: torch.Tensor, packed, *, token_offset: int,
    token_group_size: int = 32, dim_group_size: int = 32,
) -> None:
    if keys.ndim != 3 or not keys.is_cuda or not keys.is_contiguous():
        raise ValueError("keys must be contiguous CUDA [KV heads, tokens, head dim]")
    token_group_size = _validate_token_group_size(token_group_size)
    dim_group_size = _validate_dim_group_size(dim_group_size)
    if dim_group_size != 32 and token_group_size not in SPECIALIZED_TOKEN_GROUP_SIZES:
        raise ValueError("non-specialized token group sizes require dim_group_size=32")
    if dim_group_size == 32:
        _extension().pack_into(keys, *packed, int(token_offset), token_group_size)
    else:
        _extension().pack_into_dim(
            keys, *packed, int(token_offset), dim_group_size, token_group_size,
        )


def update_quest_metadata(active_keys: torch.Tensor, page_min: torch.Tensor, page_max: torch.Tensor, *, page: int) -> None:
    if active_keys.ndim != 3 or not active_keys.is_cuda or not active_keys.is_contiguous():
        raise ValueError("active_keys must be contiguous CUDA [KV heads, tokens, head dim]")
    _extension().quest_update(active_keys, page_min, page_max, int(page))


def update_fier_metadata(active_keys: torch.Tensor, packed: torch.Tensor, group_min: torch.Tensor,
                         group_max: torch.Tensor, *, token_offset: int) -> None:
    if active_keys.ndim != 3 or not active_keys.is_cuda or not active_keys.is_contiguous():
        raise ValueError("active_keys must be contiguous CUDA [KV heads, tokens, head dim]")
    _extension().fier_update(active_keys, packed, group_min, group_max, int(token_offset))


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
    dim_group_size: int = 32, dim_kernel: str | None = None,
) -> torch.Tensor:
    """Return Pop-KV approximate QK scores for every query head and token."""
    if queries.ndim != 3 or not queries.is_cuda or not queries.is_contiguous():
        raise ValueError("queries must be contiguous CUDA [batch, query heads, head dim]")
    dim_group_size = _validate_dim_group_size(dim_group_size)
    dim_kernel = _validate_dim_kernel(dim_group_size, dim_kernel)
    if dim_group_size == 32:
        return _extension().score(queries, *packed, head_to_kv.contiguous(), int(tokens))
    return _extension().score_dim(
        queries, *packed, head_to_kv.contiguous(), int(tokens), dim_group_size,
        dim_kernel == "g64-u64",
    )
