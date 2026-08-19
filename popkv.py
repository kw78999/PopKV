"""PopKV and the FullAttention, QUEST, and FIER comparison paths.

This module contains the optimized implementations used by the paper evaluations.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

METHODS = ("full", "quest", "fier", "popkv")
TOKEN_GROUP_SIZE = 32
SPECIALIZED_TOKEN_GROUP_SIZES = (8, 16, 32, 64, 128)
MAX_TOKEN_GROUP_SIZE = 4096
SUPPORTED_TOKEN_GROUP_SIZES = SPECIALIZED_TOKEN_GROUP_SIZES
PACKING_WORD_BITS = 32
SUPPORTED_DIM_GROUP_SIZES = (16, 32, 64)
SUPPORTED_DIM_KERNELS = ("auto", "g16", "g32", "g64-u32", "g64-u64")


@dataclass(frozen=True)
class RetrievalConfig:
    budget: int = 4096
    group_size: int = 32  # token-axis magnitude-threshold group (B_token)
    dim_group_size: int = 32  # dimensions sharing low/delta representatives (G_dim)
    dim_kernel: str = "auto"  # specialized score kernel; auto follows G_dim
    quest_page_size: int = 16
    full_layers: tuple[int, ...] = (0, 1)

    def validate(self) -> None:
        if self.budget <= 1:
            raise ValueError("budget must reserve at least one retrieved and one current token")
        if not 1 <= self.group_size <= MAX_TOKEN_GROUP_SIZE:
            raise ValueError(f"group_size must be between 1 and {MAX_TOKEN_GROUP_SIZE}")
        if self.dim_group_size != 32 and self.group_size not in SPECIALIZED_TOKEN_GROUP_SIZES:
            raise ValueError("non-specialized token group sizes require dim_group_size=32")
        if self.dim_group_size not in SUPPORTED_DIM_GROUP_SIZES:
            raise ValueError("dim_group_size must be one of 16, 32, 64")
        if self.dim_kernel not in SUPPORTED_DIM_KERNELS:
            raise ValueError(f"dim_kernel must be one of {SUPPORTED_DIM_KERNELS}")
        if self.dim_kernel != "auto":
            expected = 64 if self.dim_kernel.startswith("g64-") else int(self.dim_kernel[1:])
            if expected != self.dim_group_size:
                raise ValueError(
                    f"dim_kernel={self.dim_kernel} requires dim_group_size={expected}"
                )
        if self.quest_page_size <= 0:
            raise ValueError("quest_page_size must be positive")


@dataclass
class QuestCache:
    page_size: int
    token_count: int
    sealed_pages: int
    page_min: torch.Tensor
    page_max: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class FIERCache:
    group_size: int
    token_count: int
    sealed_tokens: int
    packed: torch.Tensor
    group_min: torch.Tensor
    group_max: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class PopKVCache:
    group_size: int
    dim_group_size: int
    token_count: int
    sealed_tokens: int
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    active_keys: torch.Tensor


@dataclass
class RetrievalCaches:
    method: str
    layers: dict[int, QuestCache | FIERCache | PopKVCache]


@dataclass
class StaticKVCache:
    key_storage: list[torch.Tensor]
    value_storage: list[torch.Tensor]
    length: int
    capacity: int


@dataclass
class DecodeDiagnostics:
    selected_tokens: int = 0
    available_tokens: int = 0
    sparse_head_calls: int = 0
    candidate_search_ms: float = 0.0
    retrieval_stage_ms: float = 0.0
    candidate_score_ms: float = 0.0
    candidate_topk_ms: float = 0.0
    selected_gather_ms: float = 0.0
    selected_attention_ms: float = 0.0
    frontend_ms: float = 0.0
    qkv_rope_ms: float = 0.0
    output_projection_ms: float = 0.0
    mlp_ms: float = 0.0
    final_norm_lm_head_ms: float = 0.0
    gpu_decode_ms: float = 0.0
    gpu_component_sum_ms: float = 0.0
    gpu_unattributed_ms: float = 0.0

    @property
    def candidate_ratio(self) -> float:
        return 1.0 if self.available_tokens == 0 else self.selected_tokens / self.available_tokens


_DECODE_STAGE_FIELDS = {
    "frontend": "frontend_ms",
    "qkv_rope": "qkv_rope_ms",
    "retrieval_stage": "retrieval_stage_ms",
    "score": "candidate_score_ms",
    "topk": "candidate_topk_ms",
    "gather": "selected_gather_ms",
    "attention": "selected_attention_ms",
    "output_projection": "output_projection_ms",
    "mlp": "mlp_ms",
    "final_norm_lm_head": "final_norm_lm_head_ms",
}


class _DecodeStageTimer:
    """Measure mutually exclusive decode stages without intermediate synchronization."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.segments: list[tuple[str, Any, Any]] = []
        if self.cuda:
            self.total_start = torch.cuda.Event(enable_timing=True)
            self.last = torch.cuda.Event(enable_timing=True)
            self.total_start.record(); self.last.record()
        else:
            self.total_start = self.last = time.perf_counter()

    def mark(self, name: str) -> None:
        end = torch.cuda.Event(enable_timing=True) if self.cuda else time.perf_counter()
        if self.cuda: end.record()
        self.segments.append((name, self.last, end)); self.last = end

    def finish(self, diagnostics: DecodeDiagnostics) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)
            elapsed = lambda start, end: float(start.elapsed_time(end))
        else:
            elapsed = lambda start, end: (end - start) * 1000.0
        totals = {name: 0.0 for name in _DECODE_STAGE_FIELDS}
        for name, start, end in self.segments: totals[name] += elapsed(start, end)
        for name, field_name in _DECODE_STAGE_FIELDS.items(): setattr(diagnostics, field_name, totals[name])
        diagnostics.candidate_search_ms = (diagnostics.retrieval_stage_ms +
            diagnostics.candidate_score_ms + diagnostics.candidate_topk_ms)
        diagnostics.gpu_component_sum_ms = sum(totals.values())
        diagnostics.gpu_decode_ms = elapsed(self.total_start, self.last)
        diagnostics.gpu_unattributed_ms = diagnostics.gpu_decode_ms - diagnostics.gpu_component_sum_ms


_RETRIEVAL_UPDATE_STAGES = (
    "active_key_staging", "k_metadata_encoding", "metadata_write", "group_sealing",
)


class RetrievalUpdateTimer:
    """Record mutually exclusive GPU intervals across all sparse layers."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.segments: list[tuple[str, Any, Any]] = []
        self.last = torch.cuda.Event(enable_timing=True) if self.cuda else time.perf_counter()
        if self.cuda: self.last.record()

    def mark(self, name: str) -> None:
        if name not in _RETRIEVAL_UPDATE_STAGES:
            raise ValueError(f"unknown retrieval update stage: {name}")
        end = torch.cuda.Event(enable_timing=True) if self.cuda else time.perf_counter()
        if self.cuda: end.record()
        self.segments.append((name, self.last, end)); self.last = end

    def totals(self) -> dict[str, float]:
        result = {name: 0.0 for name in _RETRIEVAL_UPDATE_STAGES}
        for name, start, end in self.segments:
            elapsed = (float(start.elapsed_time(end)) if self.cuda else (end - start) * 1000.0)
            result[name] += elapsed
        return result


def load_hf_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attention: str = "sdpa",
    token: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a gated Hugging Face Llama model after `hf auth login` or HF_TOKEN."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except Exception:
                token = None
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, use_fast=True, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        dtype=model_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attention,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def _sparse_layer_indices(num_layers: int, full_layers: Sequence[int]) -> tuple[int, ...]:
    return tuple(layer for layer in range(num_layers) if layer not in full_layers)


def _append_current(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    current = torch.full(
        (*indices.shape[:-1], 1), num_tokens - 1, device=indices.device, dtype=torch.long,
    )
    return torch.cat([indices.long(), current], dim=-1)


# ---- Quest -----------------------------------------------------------------

def build_quest_cache(keys: torch.Tensor, *, page_size: int, reserve_tokens: int) -> QuestCache:
    kv_heads, tokens, dim = (int(x) for x in keys.shape)
    sealed_pages = tokens // page_size
    sealed_tokens = sealed_pages * page_size
    capacity_pages = math.ceil((tokens + reserve_tokens + 1) / page_size)
    page_min = torch.empty((kv_heads, capacity_pages, dim), device=keys.device, dtype=torch.float32)
    page_max = torch.empty_like(page_min)
    if sealed_pages:
        pages = keys[:, :sealed_tokens].float().view(kv_heads, sealed_pages, page_size, dim)
        page_min[:, :sealed_pages].copy_(pages.amin(dim=2))
        page_max[:, :sealed_pages].copy_(pages.amax(dim=2))
    active = keys[:, sealed_tokens:].detach().clone().contiguous()
    cache = QuestCache(page_size, tokens, sealed_pages, page_min, page_max, active)
    if int(active.shape[1]):
        if active.is_cuda:
            from popkv_cuda import update_quest_metadata
            update_quest_metadata(active, page_min, page_max, page=sealed_pages)
        else:
            active_float = active.float()
            page_min[:, sealed_pages].copy_(active_float.amin(dim=1))
            page_max[:, sealed_pages].copy_(active_float.amax(dim=1))
    return cache


def append_quest(
    cache: QuestCache, keys: torch.Tensor, timer: RetrievalUpdateTimer | None = None,
) -> None:
    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    if timer: timer.mark("active_key_staging")
    cache.token_count += int(keys.shape[1])
    page = cache.sealed_pages
    if cache.active_keys.is_cuda:
        from popkv_cuda import update_quest_metadata
        update_quest_metadata(cache.active_keys, cache.page_min, cache.page_max, page=page)
    else:
        active = cache.active_keys.float()
        cache.page_min[:, page].copy_(active.amin(dim=1))
        cache.page_max[:, page].copy_(active.amax(dim=1))
    if timer:
        timer.mark("k_metadata_encoding")
        timer.mark("metadata_write")  # Stores are fused into the encoding kernel.
    if int(cache.active_keys.shape[1]) == cache.page_size:
        cache.sealed_pages += 1
        cache.active_keys = cache.active_keys[:, :0].contiguous()
    if timer: timer.mark("group_sealing")


# ---- FIER ------------------------------------------------------------------

def build_fier_cache(keys: torch.Tensor, *, group_size: int, reserve_tokens: int) -> FIERCache:
    from fier_triton import pack_keys

    packed, mins, maxs = pack_keys(keys.contiguous(), group_size=group_size)
    kv_heads, tokens, dim = (int(x) for x in keys.shape)
    capacity = math.ceil((tokens + reserve_tokens) / group_size) * group_size
    packed_storage = torch.empty(
        (kv_heads, dim, math.ceil(capacity / 32)), device=keys.device, dtype=packed.dtype,
    )
    min_storage = torch.empty(
        (kv_heads, math.ceil(capacity / group_size), dim), device=keys.device, dtype=mins.dtype,
    )
    max_storage = torch.empty_like(min_storage)
    packed_storage[:, :, : packed.shape[2]].copy_(packed)
    min_storage[:, : mins.shape[1]].copy_(mins)
    max_storage[:, : maxs.shape[1]].copy_(maxs)
    sealed = (tokens // group_size) * group_size
    cache = FIERCache(
        group_size, tokens, sealed, packed_storage, min_storage, max_storage,
        keys[:, sealed:].detach().clone().contiguous(),
    )
    if int(cache.active_keys.shape[1]): write_fier_active(cache)
    return cache


def write_fier_active(cache: FIERCache) -> None:
    from popkv_cuda import update_fier_metadata

    update_fier_metadata(
        cache.active_keys, cache.packed, cache.group_min, cache.group_max,
        token_offset=cache.sealed_tokens,
    )


def append_fier(
    cache: FIERCache, keys: torch.Tensor, timer: RetrievalUpdateTimer | None = None,
) -> None:
    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    if timer: timer.mark("active_key_staging")
    cache.token_count += int(keys.shape[1])
    write_fier_active(cache)
    if timer:
        timer.mark("k_metadata_encoding")
        timer.mark("metadata_write")  # Stores are fused into the encoding kernel.
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()
    if timer: timer.mark("group_sealing")


# ---- Pop-KV ----------------------------------------------------------------

def build_popkv_cache(
    keys: torch.Tensor, *, group_size: int, dim_group_size: int = 32, reserve_tokens: int,
) -> PopKVCache:
    from popkv_cuda import pack_keys

    if not 1 <= group_size <= MAX_TOKEN_GROUP_SIZE:
        raise ValueError(f"Pop-KV group_size must be between 1 and {MAX_TOKEN_GROUP_SIZE}")
    if dim_group_size != 32 and group_size not in SPECIALIZED_TOKEN_GROUP_SIZES:
        raise ValueError("non-specialized token group sizes require dim_group_size=32")
    if dim_group_size not in (16, 32, 64):
        raise ValueError("dim_group_size must be one of 16, 32, 64")
    tokens = int(keys.shape[1])
    capacity = math.ceil((tokens + reserve_tokens) / group_size) * group_size
    packed = pack_keys(keys.contiguous(), token_capacity=capacity,
                       token_group_size=group_size, dim_group_size=dim_group_size)
    sealed = (tokens // group_size) * group_size
    return PopKVCache(
        group_size, dim_group_size, tokens, sealed, packed,
        keys[:, sealed:].detach().clone().contiguous(),
    )


def append_popkv(
    cache: PopKVCache, keys: torch.Tensor, timer: RetrievalUpdateTimer | None = None,
) -> None:
    from popkv_cuda import pack_keys_into

    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    if timer: timer.mark("active_key_staging")
    if cache.sealed_tokens + int(cache.active_keys.shape[1]) > int(cache.packed[0].shape[1]):
        raise RuntimeError("Pop-KV cache reserve exhausted")
    pack_keys_into(
        cache.active_keys, cache.packed, token_offset=cache.sealed_tokens,
        token_group_size=cache.group_size,
        dim_group_size=cache.dim_group_size,
    )
    if timer:
        timer.mark("k_metadata_encoding")
        timer.mark("metadata_write")  # Stores are fused into the encoding kernel.
    cache.token_count += 1
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()
    if timer: timer.mark("group_sealing")


def _reference_pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a boolean final dimension into fixed-width uint32 bitplanes."""
    words = math.ceil(bits.shape[-1] / 32)
    packed = torch.zeros((*bits.shape[:-1], words), device=bits.device, dtype=torch.int32)
    for word in range(words):
        start, end = word * 32, min((word + 1) * 32, bits.shape[-1])
        lanes = torch.arange(end - start, device=bits.device, dtype=torch.int64)
        weights = torch.bitwise_left_shift(torch.ones_like(lanes), lanes)
        packed[..., word] = (bits[..., start:end].to(torch.int64) * weights).sum(-1).to(torch.int32)
    return packed


def _reference_representatives(
    values: torch.Tensor, magnitude: torch.Tensor, dim_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = math.ceil(values.shape[-1] / dim_group_size)
    # Preserve the release G32 reduction shape/order exactly for regression tests.
    if dim_group_size == 32:
        pad = groups * 32 - values.shape[-1]
        absolute, mag, valid = values.abs(), magnitude, torch.ones_like(magnitude)
        if pad:
            shape = (*values.shape[:-1], pad)
            absolute = torch.cat([absolute, torch.zeros(shape, device=values.device)], -1)
            mag = torch.cat([mag, torch.zeros(shape, dtype=torch.bool, device=values.device)], -1)
            valid = torch.cat([valid, torch.zeros(shape, dtype=torch.bool, device=values.device)], -1)
        shape = (*values.shape[:-1], groups, 32)
        absolute, mag, valid = absolute.view(shape), mag.view(shape), valid.view(shape)
        low_mask, high_mask = (~mag) & valid, mag & valid
        low_count, high_count = low_mask.sum(-1), high_mask.sum(-1)
        low = (absolute * low_mask).sum(-1) / low_count.clamp_min(1)
        high = (absolute * high_mask).sum(-1) / high_count.clamp_min(1)
        low = torch.where(low_count > 0, low, high)
        high = torch.where(high_count > 0, high, low)
        return low, high - low
    low_out, delta_out = [], []
    for group in range(groups):
        start, end = group * dim_group_size, min((group + 1) * dim_group_size, values.shape[-1])
        absolute = values[..., start:end].abs()
        mag = magnitude[..., start:end]
        low_mask, high_mask = ~mag, mag
        low_count, high_count = low_mask.sum(-1), high_mask.sum(-1)
        low_sum = (absolute * low_mask).sum(-1)
        high_sum = (absolute * high_mask).sum(-1)
        low = torch.where(
            low_count > 0, low_sum / low_count.clamp_min(1),
            torch.where(high_count > 0, high_sum / high_count.clamp_min(1), torch.zeros_like(low_sum)),
        )
        high = torch.where(high_count > 0, high_sum / high_count.clamp_min(1), low)
        low_out.append(low); delta_out.append(high - low)
    return torch.stack(low_out, -1), torch.stack(delta_out, -1)


def reference_query_encode(
    queries: torch.Tensor, dim_group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Readable reference separating 32-bit packing from representative groups.

    Sign/magnitude have ceil(D/32) words while low/delta have ceil(D/G_dim)
    groups. Query magnitude thresholds continue to use the whole query head.
    """
    if queries.ndim != 3:
        raise ValueError("queries must be [batch, query heads, head dim]")
    if queries.shape[-1] <= 0:
        raise ValueError("query head dimension must be non-empty")

    if dim_group_size not in (16, 32, 64):
        raise ValueError("dim_group_size must be one of 16, 32, 64")
    q = queries.float()
    minimum = q.amin(dim=-1, keepdim=True)
    maximum = q.amax(dim=-1, keepdim=True)
    sign = q >= 0
    magnitude = torch.where(sign, q > maximum * 0.5, q < minimum * 0.5)

    low, delta = _reference_representatives(q, magnitude, dim_group_size)
    return _reference_pack_bits(sign), _reference_pack_bits(magnitude), low, delta


def reference_query_encode_g32(
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper for the release G_dim=32 reference."""
    return reference_query_encode(queries, 32)


def _reference_query_encode_g32_oracle(
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen release-semantics oracle used only by G32 regression tests."""
    q = queries.float()
    qmin, qmax = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    sign = q >= 0
    magnitude = torch.where(sign, q > qmax * 0.5, q < qmin * 0.5)
    words = math.ceil(q.shape[-1] / 32)
    pad = words * 32 - q.shape[-1]
    absolute, valid, mag = q.abs(), torch.ones_like(magnitude), magnitude
    if pad:
        shape = (*q.shape[:-1], pad)
        absolute = torch.cat([absolute, torch.zeros(shape, device=q.device)], -1)
        valid = torch.cat([valid, torch.zeros(shape, dtype=torch.bool, device=q.device)], -1)
        mag = torch.cat([mag, torch.zeros(shape, dtype=torch.bool, device=q.device)], -1)
    shape = (*q.shape[:-1], words, 32)
    absolute, valid, mag = absolute.view(shape), valid.view(shape), mag.view(shape)
    low_mask, high_mask = (~mag) & valid, mag & valid
    low_count, high_count = low_mask.sum(-1), high_mask.sum(-1)
    low = (absolute * low_mask).sum(-1) / low_count.clamp_min(1)
    high = (absolute * high_mask).sum(-1) / high_count.clamp_min(1)
    low = torch.where(low_count > 0, low, high)
    high = torch.where(high_count > 0, high, low)
    return _reference_pack_bits(sign), _reference_pack_bits(magnitude), low, high - low


def reference_key_encode(
    keys: torch.Tensor, *, dim_group_size: int = 32, token_group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference key encoder with fixed uint32 packing and explicit axis groups."""
    if keys.ndim != 3 or keys.shape[-1] <= 0 or keys.shape[-2] <= 0:
        raise ValueError("keys must be [KV heads, tokens, non-empty head dim]")
    if not 1 <= token_group_size <= MAX_TOKEN_GROUP_SIZE:
        raise ValueError(f"token_group_size must be between 1 and {MAX_TOKEN_GROUP_SIZE}")
    if dim_group_size not in (16, 32, 64):
        raise ValueError("dim_group_size must be one of 16, 32, 64")
    k = keys.float()
    kv_heads, tokens, dim = k.shape
    token_groups = math.ceil(tokens / token_group_size)
    padded = token_groups * token_group_size - tokens
    extrema = F.pad(k, (0, 0, 0, padded), value=float("nan")) if padded else k
    grouped = extrema.view(kv_heads, token_groups, token_group_size, dim)
    minimum = torch.nan_to_num(grouped, nan=float("inf")).amin(2)
    maximum = torch.nan_to_num(grouped, nan=-float("inf")).amax(2)
    group_ids = torch.arange(tokens, device=k.device) // token_group_size
    minimum, maximum = minimum[:, group_ids], maximum[:, group_ids]
    sign = k >= 0
    magnitude = torch.where(sign, k > maximum * 0.5, k < minimum * 0.5)
    low, delta = _reference_representatives(k, magnitude, dim_group_size)
    return _reference_pack_bits(sign), _reference_pack_bits(magnitude), low, delta


def _popkv_reference_scores_g32_oracle(
    queries: torch.Tensor, keys: torch.Tensor, *, head_to_kv: torch.Tensor, group_size: int = 32,
) -> torch.Tensor:
    """Readable reference for the final Pop-KV score used by correctness tests."""
    def representatives(x: torch.Tensor, magnitude: torch.Tensor):
        words = math.ceil(x.shape[-1] / 32)
        pad = words * 32 - x.shape[-1]
        absolute = x.abs()
        valid = torch.ones_like(magnitude)
        if pad:
            shape = (*x.shape[:-1], pad)
            magnitude = torch.cat([magnitude, torch.zeros(shape, dtype=torch.bool, device=x.device)], -1)
            valid = torch.cat([valid, torch.zeros(shape, dtype=torch.bool, device=x.device)], -1)
            absolute = torch.cat([absolute, torch.zeros(shape, device=x.device)], -1)
        shape = (*magnitude.shape[:-1], words, 32)
        mag, valid, absolute = magnitude.view(shape), valid.view(shape), absolute.view(shape)
        low_mask, high_mask = (~mag) & valid, mag & valid
        low_count, high_count = low_mask.sum(-1), high_mask.sum(-1)
        low = (absolute * low_mask).sum(-1) / low_count.clamp_min(1)
        high = (absolute * high_mask).sum(-1) / high_count.clamp_min(1)
        low = torch.where(low_count > 0, low, high)
        high = torch.where(high_count > 0, high, low)
        return low, high

    q = queries.float()
    qmin, qmax = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    qsign = q >= 0
    qmag = torch.where(qsign, q > qmax * .5, q < qmin * .5)
    qlow, qhigh = representatives(q, qmag)

    k = keys.float()
    kvh, tokens, dim = k.shape
    groups = math.ceil(tokens / group_size)
    padded = groups * group_size - tokens
    extrema = F.pad(k, (0, 0, 0, padded), value=float("nan")) if padded else k
    grouped = extrema.view(kvh, groups, group_size, dim)
    minimum = torch.nan_to_num(grouped, nan=float("inf")).amin(2)
    maximum = torch.nan_to_num(grouped, nan=-float("inf")).amax(2)
    group_ids = torch.arange(tokens, device=k.device) // group_size
    minimum, maximum = minimum[:, group_ids], maximum[:, group_ids]
    ksign = k >= 0
    kmag = torch.where(ksign, k > maximum * .5, k < minimum * .5)
    klow, khigh = representatives(k, kmag)

    words = qlow.shape[-1]
    ql = qlow.repeat_interleave(32, -1)[..., :dim]
    qh = qhigh.repeat_interleave(32, -1)[..., :dim]
    kl = klow.index_select(0, head_to_kv).repeat_interleave(32, -1)[..., :dim]
    kh = khigh.index_select(0, head_to_kv).repeat_interleave(32, -1)[..., :dim]
    qhat = torch.where(qsign, 1.0, -1.0) * torch.where(qmag, qh, ql)
    mapped_sign = ksign.index_select(0, head_to_kv)
    mapped_mag = kmag.index_select(0, head_to_kv)
    khat = torch.where(mapped_sign, 1.0, -1.0) * torch.where(mapped_mag, kh, kl)
    return torch.einsum("bhd,htd->bht", qhat, khat) / math.sqrt(dim)


def popkv_reference_scores(
    queries: torch.Tensor, keys: torch.Tensor, *, head_to_kv: torch.Tensor,
    group_size: int = 32, dim_group_size: int = 32,
) -> torch.Tensor:
    """Readable reconstructed PopKV score for G_dim in {16, 32, 64}."""
    q = queries.float(); k = keys.float(); dim = int(q.shape[-1])
    q_sign_bits, q_mag_bits, q_low, q_delta = reference_query_encode(q, dim_group_size)
    k_sign_bits, k_mag_bits, k_low, k_delta = reference_key_encode(
        k, dim_group_size=dim_group_size, token_group_size=group_size,
    )
    del q_sign_bits, q_mag_bits, k_sign_bits, k_mag_bits  # packed layouts are tested separately
    qmin, qmax = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    qsign = q >= 0
    qmag = torch.where(qsign, q > qmax * .5, q < qmin * .5)
    token_groups = math.ceil(k.shape[1] / group_size)
    padded = token_groups * group_size - k.shape[1]
    extrema = F.pad(k, (0, 0, 0, padded), value=float("nan")) if padded else k
    grouped = extrema.view(k.shape[0], token_groups, group_size, dim)
    minimum = torch.nan_to_num(grouped, nan=float("inf")).amin(2)
    maximum = torch.nan_to_num(grouped, nan=-float("inf")).amax(2)
    group_ids = torch.arange(k.shape[1], device=k.device) // group_size
    minimum, maximum = minimum[:, group_ids], maximum[:, group_ids]
    ksign = k >= 0
    kmag = torch.where(ksign, k > maximum * .5, k < minimum * .5)
    ql = q_low.repeat_interleave(dim_group_size, -1)[..., :dim]
    qh = (q_low + q_delta).repeat_interleave(dim_group_size, -1)[..., :dim]
    klow = k_low.index_select(0, head_to_kv)
    kdelta = k_delta.index_select(0, head_to_kv)
    kl = klow.repeat_interleave(dim_group_size, -1)[..., :dim]
    kh = (klow + kdelta).repeat_interleave(dim_group_size, -1)[..., :dim]
    qhat = torch.where(qsign, 1.0, -1.0) * torch.where(qmag, qh, ql)
    mapped_sign = ksign.index_select(0, head_to_kv)
    mapped_mag = kmag.index_select(0, head_to_kv)
    khat = torch.where(mapped_sign, 1.0, -1.0) * torch.where(mapped_mag, kh, kl)
    return torch.einsum("bhd,htd->bht", qhat, khat) / math.sqrt(dim)


def popkv_reference_bitwise_scores(
    queries: torch.Tensor, keys: torch.Tensor, *, head_to_kv: torch.Tensor,
    group_size: int = 32, dim_group_size: int = 32,
) -> torch.Tensor:
    """Slow CPU-oriented popcount decomposition used to test G_dim semantics."""
    q_sign, q_mag, q_low, q_delta = reference_query_encode(queries, dim_group_size)
    k_sign, k_mag, k_low, k_delta = reference_key_encode(
        keys, dim_group_size=dim_group_size, token_group_size=group_size,
    )
    batch, q_heads, dim = queries.shape
    tokens = keys.shape[1]
    rep_groups = math.ceil(dim / dim_group_size)
    output = torch.empty((batch, q_heads, tokens), dtype=torch.float32, device=queries.device)
    full_word = (1 << 32) - 1
    for b in range(batch):
        for qh in range(q_heads):
            kvh = int(head_to_kv[qh])
            for token in range(tokens):
                total = 0.0
                for rep in range(rep_groups):
                    start = rep * dim_group_size
                    end = min(start + dim_group_size, dim)
                    counts = [0, 0, 0, 0]
                    for word in range(start // 32, (end - 1) // 32 + 1):
                        word_start = word * 32
                        begin = max(start, word_start) - word_start
                        finish = min(end, word_start + 32) - word_start
                        width = finish - begin
                        mask = (full_word if width == 32 else (1 << width) - 1) << begin
                        qs = int(q_sign[b, qh, word]) & full_word
                        qm = int(q_mag[b, qh, word]) & mask
                        ks = int(k_sign[kvh, token, word]) & full_word
                        km = int(k_mag[kvh, token, word]) & mask
                        agreement = (~(qs ^ ks)) & mask
                        both = qm & km
                        selections = (mask, qm, km, both)
                        for index, selection in enumerate(selections):
                            counts[index] += 2 * (agreement & selection).bit_count() - selection.bit_count()
                    ql, dq = float(q_low[b, qh, rep]), float(q_delta[b, qh, rep])
                    kl, dk = float(k_low[kvh, token, rep]), float(k_delta[kvh, token, rep])
                    total += (ql * kl * counts[0] + dq * kl * counts[1] +
                              ql * dk * counts[2] + dq * dk * counts[3])
                output[b, qh, token] = total / math.sqrt(dim)
    return output


# ---- Shared attention and model adapter ------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _layer_cache(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, StaticKVCache):
        return cache.key_storage[layer][..., :cache.length, :], cache.value_storage[layer][..., :cache.length, :]
    try:
        return cache[layer][:2]
    except (TypeError, IndexError, KeyError):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return cache.key_cache[layer], cache.value_cache[layer]
        raise TypeError(f"unsupported Transformers cache: {type(cache).__name__}")


class LlamaSparseDecoder:
    """One-token Llama decoder exposing FullAttention, Quest, FIER, and Pop-KV."""

    def __init__(self, model: Any, retrieval: RetrievalConfig = RetrievalConfig()) -> None:
        retrieval.validate()
        self.model = model
        self.backbone = getattr(model, "model", None)
        if self.backbone is None:
            raise TypeError("expected a LlamaForCausalLM-compatible model")
        self.layers = self.backbone.layers
        self.retrieval = retrieval
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(getattr(model.config, "num_key_value_heads", self.num_heads))
        self.kv_group_size = self.num_heads // self.num_kv_heads
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // self.num_heads))
        self.head_to_kv = torch.arange(self.num_heads, device=self.device, dtype=torch.long) // self.kv_group_size

    @property
    def device(self) -> torch.device:
        return self.backbone.embed_tokens.weight.device

    @torch.inference_mode()
    def build_prefix_cache(self, prefix_ids: torch.Tensor):
        ids = prefix_ids.to(self.device).view(1, -1)
        return self.backbone(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
            output_hidden_states=False, return_dict=True,
        ).past_key_values

    @torch.inference_mode()
    def build_retrieval_caches(
        self, prefix_cache: Any, *, method: str, reserve_tokens: int = 2048,
    ) -> RetrievalCaches:
        if method not in {"quest", "fier", "popkv"}:
            raise ValueError("retrieval caches are only used by sparse methods")
        layers: dict[int, QuestCache | FIERCache | PopKVCache] = {}
        for layer in _sparse_layer_indices(len(self.layers), self.retrieval.full_layers):
            keys, _ = _layer_cache(prefix_cache, layer)
            keys = keys[0].to(self.device)
            if method == "quest":
                layers[layer] = build_quest_cache(
                    keys, page_size=self.retrieval.quest_page_size, reserve_tokens=reserve_tokens,
                )
            elif method == "fier":
                layers[layer] = build_fier_cache(
                    keys, group_size=TOKEN_GROUP_SIZE, reserve_tokens=reserve_tokens,
                )
            else:
                layers[layer] = build_popkv_cache(
                    keys, group_size=self.retrieval.group_size,
                    dim_group_size=self.retrieval.dim_group_size, reserve_tokens=reserve_tokens,
                )
        return RetrievalCaches(method, layers)

    @torch.inference_mode()
    def append_retrieval_caches(
        self, caches: RetrievalCaches, new_keys: Sequence[torch.Tensor], *, measure: bool = False,
    ) -> RetrievalUpdateTimer | None:
        if not measure:
            for layer, keys in enumerate(new_keys):
                cache = caches.layers.get(layer)
                if cache is None: continue
                if isinstance(cache, QuestCache): append_quest(cache, keys[0])
                elif isinstance(cache, FIERCache): append_fier(cache, keys[0])
                elif isinstance(cache, PopKVCache): append_popkv(cache, keys[0])
                else: raise TypeError("unknown retrieval cache")
            return None

        # Batch equal phases across layers so measurement needs four events per
        # token rather than four events per layer. Operations remain ordered on
        # the same CUDA stream and retain the same cache semantics.
        timer = RetrievalUpdateTimer(self.device)
        items = [(cache, new_keys[layer][0]) for layer, cache in caches.layers.items()]
        for cache, keys in items:
            cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
            if isinstance(cache, QuestCache): cache.token_count += int(keys.shape[1])
            elif isinstance(cache, FIERCache): cache.token_count += int(keys.shape[1])
            elif isinstance(cache, PopKVCache):
                if cache.sealed_tokens + int(cache.active_keys.shape[1]) > int(cache.packed[0].shape[1]):
                    raise RuntimeError("Pop-KV cache reserve exhausted")
                cache.token_count += 1
            else: raise TypeError("unknown retrieval cache")
        timer.mark("active_key_staging")

        for cache, _ in items:
            if isinstance(cache, QuestCache):
                if cache.active_keys.is_cuda:
                    from popkv_cuda import update_quest_metadata
                    update_quest_metadata(
                        cache.active_keys, cache.page_min, cache.page_max, page=cache.sealed_pages,
                    )
                else:
                    active = cache.active_keys.float()
                    cache.page_min[:, cache.sealed_pages].copy_(active.amin(dim=1))
                    cache.page_max[:, cache.sealed_pages].copy_(active.amax(dim=1))
            elif isinstance(cache, FIERCache): write_fier_active(cache)
            else:
                from popkv_cuda import pack_keys_into
                pack_keys_into(
                    cache.active_keys, cache.packed, token_offset=cache.sealed_tokens,
                    token_group_size=cache.group_size,
                    dim_group_size=cache.dim_group_size,
                )
        timer.mark("k_metadata_encoding")
        timer.mark("metadata_write")  # Direct stores are fused into the kernels above.

        for cache, _ in items:
            group_size = cache.page_size if isinstance(cache, QuestCache) else cache.group_size
            if int(cache.active_keys.shape[1]) == group_size:
                if isinstance(cache, QuestCache): cache.sealed_pages += 1
                else: cache.sealed_tokens += group_size
                cache.active_keys = cache.active_keys[:, :0].contiguous()
        timer.mark("group_sealing")
        return timer

    @torch.inference_mode()
    def build_static_kv_cache(self, prefix_cache: Any, *, reserve_tokens: int) -> StaticKVCache:
        first, _ = _layer_cache(prefix_cache, 0)
        length, capacity = int(first.shape[-2]), int(first.shape[-2]) + reserve_tokens
        keys_out, values_out = [], []
        for layer in range(len(self.layers)):
            keys, values = _layer_cache(prefix_cache, layer)
            key_store = torch.empty((*keys.shape[:-2], capacity, keys.shape[-1]), device=keys.device, dtype=keys.dtype)
            value_store = torch.empty_like(key_store)
            key_store[..., :length, :].copy_(keys); value_store[..., :length, :].copy_(values)
            keys_out.append(key_store); values_out.append(value_store)
        return StaticKVCache(keys_out, values_out, length, capacity)

    @torch.inference_mode()
    def append_kv(self, cache: StaticKVCache, new_keys: Sequence[torch.Tensor], new_values: Sequence[torch.Tensor]) -> None:
        if cache.length >= cache.capacity:
            raise RuntimeError("static KV cache capacity exhausted")
        position = cache.length
        for layer, (keys, values) in enumerate(zip(new_keys, new_values)):
            cache.key_storage[layer][..., position:position + 1, :].copy_(keys)
            cache.value_storage[layer][..., position:position + 1, :].copy_(values)
        cache.length += 1

    def _position_embeddings(self, hidden: torch.Tensor, positions: torch.Tensor, values: torch.Tensor):
        rotary = getattr(self.backbone, "rotary_emb", None)
        if rotary is not None:
            try: return rotary(hidden, positions)
            except TypeError: pass
        rotary = getattr(self.layers[0].self_attn, "rotary_emb", None)
        if rotary is None: raise RuntimeError("cannot find Llama rotary embedding")
        try: return rotary(values, positions)
        except TypeError:
            cos, sin = rotary(values, seq_len=int(positions.max()) + 1)
            position = positions.item()
            return cos[:, position:position + 1], sin[:, position:position + 1]

    @torch.inference_mode()
    def decode(
        self, cache: StaticKVCache, token_id: int | torch.Tensor, *, method: str,
        retrieval_caches: RetrievalCaches | None = None, return_new_kv: bool = False,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if method != "full" and retrieval_caches is None:
            raise ValueError(f"{method} requires build_retrieval_caches()")
        timer = _DecodeStageTimer(self.device)
        prefix_length = cache.length
        token = torch.as_tensor(token_id, device=self.device, dtype=torch.long).view(1, 1)
        hidden = self.backbone.embed_tokens(token)
        positions = torch.tensor([[prefix_length]], device=self.device)
        first_attn = self.layers[0].self_attn
        first_value = first_attn.v_proj(hidden).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._position_embeddings(hidden, positions, first_value)
        timer.mark("frontend")
        diag = DecodeDiagnostics()
        new_keys, new_values = [], []

        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            q = attention.q_proj(normalized).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = attention.k_proj(normalized).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attention.v_proj(normalized).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, cos.to(q.device), sin.to(q.device))
            prefix_k, prefix_v = _layer_cache(cache, layer_idx)
            if return_new_kv:
                new_keys.append(k.detach()); new_values.append(v.detach())
            timer.mark("qkv_rope")

            force_full = method == "full" or layer_idx in self.retrieval.full_layers
            if force_full:
                def gather_full(): return torch.cat([prefix_k, k], 2), torch.cat([prefix_v, v], 2)
                all_k, all_v = gather_full(); timer.mark("gather")
                def attend_full():
                    grouped = q[0].view(self.num_kv_heads, self.kv_group_size, 1, self.head_dim)
                    return F.scaled_dot_product_attention(
                        grouped, all_k[0, :, None], all_v[0, :, None], dropout_p=0.0, is_causal=False,
                    ).reshape(self.num_heads, self.head_dim)
                output = attend_full(); timer.mark("attention")
            else:
                num_tokens = prefix_length + 1
                sparse_cache = retrieval_caches.layers[layer_idx]  # type: ignore[union-attr]
                valid = None
                if method == "quest":
                    assert isinstance(sparse_cache, QuestCache)
                    from sparse_kernels import quest_indices, quest_scores
                    pages = sparse_cache.sealed_pages + int(sparse_cache.active_keys.shape[1] > 0)
                    scores = quest_scores(
                        q[0, :, 0], sparse_cache.page_min, sparse_cache.page_max,
                        self.head_to_kv, pages=pages,
                    )
                    timer.mark("score")
                    indices, valid = quest_indices(
                        scores, num_tokens=num_tokens, page_size=self.retrieval.quest_page_size,
                        budget=self.retrieval.budget,
                    )
                    timer.mark("topk")
                elif method == "fier":
                    assert isinstance(sparse_cache, FIERCache)
                    from fier_triton import score_packed_batched
                    def score_fier():
                        return score_packed_batched(
                            q[0, :, 0].unsqueeze(0), sparse_cache.packed,
                            sparse_cache.group_min, sparse_cache.group_max,
                            self.head_to_kv, tokens=prefix_length, group_size=sparse_cache.group_size,
                        )
                    scores = score_fier(); timer.mark("score")
                    def topk_fier():
                        selected = torch.topk(
                            scores, k=min(self.retrieval.budget - 1, prefix_length),
                            dim=-1, sorted=False,
                        ).indices[0]
                        return _append_current(selected, num_tokens)
                    indices = topk_fier(); timer.mark("topk")
                else:
                    assert isinstance(sparse_cache, PopKVCache)
                    from popkv_cuda import score as score_popkv
                    scores = score_popkv(
                        q[0, :, 0].unsqueeze(0).contiguous(), sparse_cache.packed,
                        head_to_kv=self.head_to_kv, tokens=sparse_cache.token_count,
                        dim_group_size=sparse_cache.dim_group_size,
                        dim_kernel=self.retrieval.dim_kernel,
                    )
                    timer.mark("score")
                    def topk_popkv():
                        selected = torch.topk(
                            scores, k=min(self.retrieval.budget - 1, prefix_length),
                            dim=-1, sorted=False,
                        ).indices[0]
                        return _append_current(selected, num_tokens)
                    indices = topk_popkv(); timer.mark("topk")

                diag.selected_tokens += int(indices.numel())
                diag.available_tokens += num_tokens * self.num_heads
                diag.sparse_head_calls += self.num_heads
                # Indexed attention reads selected K/V directly from the cache;
                # no per-query-head gathered K/V tensor is materialized.
                timer.mark("gather")
                from sparse_kernels import indexed_attention
                output = indexed_attention(
                    q[0, :, 0], prefix_k[0], prefix_v[0], k[0, :, 0], v[0, :, 0],
                    indices, self.head_to_kv, valid,
                )
                timer.mark("attention")

            output = output.reshape(1, 1, -1).to(normalized.dtype)
            hidden = residual + attention.o_proj(output)
            timer.mark("output_projection")
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
            timer.mark("mlp")

        hidden = self.backbone.norm(hidden)
        logits = self.model.lm_head(hidden).float()[0, -1]
        timer.mark("final_norm_lm_head")
        timer.finish(diag)
        if return_new_kv:
            return logits, diag, new_keys, new_values
        return logits, diag
