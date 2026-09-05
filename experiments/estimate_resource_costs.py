from __future__ import annotations

"""Deterministic operation, tensor-traffic, and cache-footprint accounting.

The accounting follows the kernels in this repository.  It is intentionally not
a claim about hardware-counter DRAM traffic: ``bytes`` is logical tensor traffic
(bytes read plus bytes written at an operator boundary).  One FMA is two FLOPs.
Comparisons, integer/bit operations, reductions such as max, and transcendental
operations are not silently converted to FLOPs.  CUDA ``__popc`` invocations are
reported independently as POpcount.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelShape:
    layers: int = 32
    hidden: int = 4096
    intermediate: int = 14336
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 128256
    dtype_bytes: int = 2


@dataclass(frozen=True)
class Profile:
    label: str
    method: str
    token_group: int = 0
    dim_group: int = 0
    page_size: int = 16
    fier_kernel: str = ""
    score_kernel: str = ""
    score_bytes: int = 4


PROFILES = (
    Profile("Full", "full"),
    Profile("QUEST", "quest", page_size=16),
    Profile("FIER", "fier", token_group=32, fier_kernel="fier", score_bytes=2),
    Profile("PopKV-Normal", "popkv", token_group=32, dim_group=32,
            score_kernel="fp16-gqa-tiled", score_bytes=2),
    Profile("PopKV-Q", "popkv", token_group=128, dim_group=16,
            score_kernel="fp16-gqa-tiled", score_bytes=2),
    Profile("PopKV-S", "popkv", token_group=64, dim_group=64,
            score_kernel="fp16-gqa-tiled", score_bytes=2),
)


@dataclass
class Cost:
    flops: int = 0
    popcount: int = 0
    bytes: int = 0
    comparisons: int = 0
    integer_ops: int = 0
    bitwise_ops: int = 0
    conversions: int = 0
    exp_ops: int = 0
    rsqrt_ops: int = 0
    topk_inputs: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(*(getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__))

    def __mul__(self, factor: int) -> "Cost":
        return Cost(*(getattr(self, name) * factor for name in self.__dataclass_fields__))


def _matvec(inputs: int, outputs: int, dtype_bytes: int) -> Cost:
    # Read input and weights, write output. Bias-free Llama projections.
    return Cost(2 * inputs * outputs, 0, dtype_bytes * (inputs + inputs * outputs + outputs))


def _norm(width: int, dtype_bytes: int) -> Cost:
    # square + reduction + scale/divide + output multiply; rsqrt is not a FLOP.
    return Cost(3 * width, 0, dtype_bytes * (2 * width) + 4 * width, rsqrt_ops=1)


def _common_stages(shape: ModelShape) -> dict[str, Cost]:
    h, i, d = shape.hidden, shape.intermediate, shape.dtype_bytes
    q_width, kv_width = shape.q_heads * shape.head_dim, shape.kv_heads * shape.head_dim
    frontend = _matvec(h, kv_width, d)  # extra first-layer V projection used to construct RoPE
    frontend.bytes += d * h  # token embedding row
    qkv = _norm(h, d) + _matvec(h, q_width, d) + _matvec(h, kv_width, d) * 2
    qkv.flops += 3 * (q_width + kv_width)  # pairwise RoPE: 4 mul + 2 add per pair
    output = _matvec(h, h, d)
    output.flops += h  # residual add
    output.bytes += d * 3 * h
    mlp = _norm(h, d) + _matvec(h, i, d) * 2 + _matvec(i, h, d)
    mlp.flops += 2 * i + h  # SiLU product, gated product, residual add; sigmoid excluded
    mlp.exp_ops += i
    mlp.bytes += d * (4 * i + 3 * h)
    final = _norm(h, d) + _matvec(h, shape.vocab, d)
    return {
        "frontend": frontend,
        "qkv_rope": qkv * shape.layers,
        "output_projection": output * shape.layers,
        "mlp": mlp * shape.layers,
        "final_norm_lm_head": final,
    }


def _attention(shape: ModelShape, tokens: int, *, valid_mask: bool = False,
               indexed: bool = False, split_tokens: int = 0) -> Cost:
    q, dim, d = shape.q_heads, shape.head_dim, shape.dtype_bytes
    # QK and PV matvecs only. Softmax exp/max are excluded; its sum and divide are included.
    flops = q * (4 * tokens * dim + max(0, tokens - 1) + tokens)
    traffic = q * (2 * tokens * dim * d + tokens * 8 + (tokens if valid_mask else 0))
    traffic += d * 2 * q * dim  # query read and output write
    comparisons = q * max(0, tokens - 1)
    integer_ops = q * tokens if indexed else 0
    if indexed:
        comparisons += q * tokens * (2 + int(valid_mask))
    exp_ops = q * tokens
    if indexed and split_tokens:
        splits = math.ceil(tokens / split_tokens)
        # Merge reads every FP32 partial state once after it was written.
        traffic += q * splits * (dim + 2) * 2 * 4
        flops += q * (dim * (2 * splits) + 3 * splits)
        comparisons += q * max(0, splits - 1)
        exp_ops += q * splits
    return Cost(flops, 0, traffic, comparisons=comparisons, integer_ops=integer_ops,
                exp_ops=exp_ops)


def _full_gather(shape: ModelShape, tokens: int) -> Cost:
    # torch.cat reads K/V and writes the materialized K/V pair.
    kv = 2 * shape.kv_heads * tokens * shape.head_dim * shape.dtype_bytes
    return Cost(bytes=2 * kv)


def _score(profile: Profile, shape: ModelShape, prefix_tokens: int) -> Cost:
    q, dim, d = shape.q_heads, shape.head_dim, shape.dtype_bytes
    if profile.method == "quest":
        pages = math.ceil(prefix_tokens / profile.page_size)
        blocks = math.ceil(pages / 16)
        flops = q * pages * (3 * dim - 1)  # 2 multiplies and dot reduction; max is a comparison
        traffic = q * (blocks * dim * d + pages * dim * 2 * 4 + pages * 4)
        return Cost(flops, 0, traffic, comparisons=q * pages * dim,
                    conversions=q * blocks * dim)
    if profile.method == "fier":
        blocks = math.ceil(prefix_tokens / 32)
        elements = q * prefix_tokens * dim
        if profile.fier_kernel == "fier":
            # Public-code structure: bit->FP, scale*bit+zero, then FP32 MAC.
            # Scores are FP16; packed words and metadata are D-contiguous.
            flops = 4 * elements
            traffic = q * (
                blocks * dim * (d + 4 + 2 * d) + prefix_tokens * 2
            )
            conversions = q * (
                blocks * dim * 3 + prefix_tokens * dim + prefix_tokens
            )
            return Cost(flops, 0, traffic, bitwise_ops=2 * elements,
                        conversions=conversions)
    if profile.method == "popkv":
        words, reps = math.ceil(dim / 32), math.ceil(dim / profile.dim_group)
        # Query is traversed for extrema, bit encoding, and representative encoding.
        query_flops = q * (dim + 2 * reps)  # magnitude thresholds and low/delta finalization
        reconstruct_flops = q * prefix_tokens * (15 * reps + 1)
        # G16 invokes the generic 8-popcount reconstruction per representative.
        # G64-u32 saves the constant mask popcount: 7 calls per 32-bit half.
        popc_per_token = 8 * reps if profile.dim_group == 16 else 7 * words
        if profile.score_kernel == "fp16-gqa-tiled":
            # Separate warp query encoding; K is loaded once per KV head and
            # reused by all GQA query heads. Query state is staged per tile.
            tiles = math.ceil(prefix_tokens / 1024)
            query_state = words * 2 * 4 + reps * 2 * 4
            traffic = (
                q * (2 * dim * d + query_state * (1 + tiles))
                + shape.kv_heads * prefix_tokens * (words * 8 + reps * 4)
                + q * prefix_tokens * profile.score_bytes
                + shape.kv_heads * tiles * 8
            )
            conversions = (
                q * 2 * dim
                + shape.kv_heads * prefix_tokens * 2 * reps
                + q * prefix_tokens * (4 * reps + 1)
            )
        else:
            traffic = q * (
                3 * dim * d + prefix_tokens * (words * 8 + reps * 4 + profile.score_bytes)
            )
            conversions = q * (3 * dim + prefix_tokens * 6 * reps)
        reconstruct = q * prefix_tokens * reps
        comparisons = q * (4 * dim - 2)
        bitwise_per_rep = 9 if profile.dim_group == 16 else 18
        return Cost(query_flops + reconstruct_flops, q * prefix_tokens * popc_per_token, traffic,
                    comparisons=comparisons, integer_ops=8 * reconstruct,
                    bitwise_ops=q * (2 * dim) + reconstruct * bitwise_per_rep,
                    conversions=conversions, rsqrt_ops=q)
    return Cost()


def _topk(profile: Profile, shape: ModelShape, prefix_tokens: int, budget: int) -> Cost:
    q = shape.q_heads
    if profile.method == "quest":
        candidates = math.ceil(prefix_tokens / profile.page_size)
        selected = min(candidates, math.ceil(budget / profile.page_size))
        output = selected * profile.page_size
        # Lower-bound tensor traffic. Comparisons in torch.topk are deliberately not FLOPs.
        return Cost(bytes=q * (candidates * profile.score_bytes + selected * 8 + output * (8 + 1)),
                    integer_ops=q * output * 3, comparisons=q * output,
                    topk_inputs=q * candidates)
    candidates = prefix_tokens
    selected = min(budget - 1, prefix_tokens)
    return Cost(bytes=q * (candidates * profile.score_bytes + selected * 8 + (selected + 1) * 8),
                integer_ops=q, topk_inputs=q * candidates)


def _kv_update(shape: ModelShape) -> Cost:
    # Read new K/V and write them into the static cache.
    return Cost(bytes=2 * 2 * shape.layers * shape.kv_heads * shape.head_dim * shape.dtype_bytes)


def _metadata_bytes_per_layer(profile: Profile, shape: ModelShape, tokens: int) -> int:
    k, dim, d = shape.kv_heads, shape.head_dim, shape.dtype_bytes
    if profile.method == "quest":
        return k * math.ceil(tokens / profile.page_size) * dim * 2 * 4
    if profile.method == "fier":
        return k * (dim * math.ceil(tokens / 32) * 4 + math.ceil(tokens / profile.token_group) * dim * 2 * d)
    if profile.method == "popkv":
        return k * tokens * (math.ceil(dim / 32) * 2 * 4 + math.ceil(dim / profile.dim_group) * 2 * 2)
    return 0


def _retrieval_update(profile: Profile, shape: ModelShape, active: int, sparse_layers: int) -> Cost:
    if profile.method == "full":
        return Cost()
    k, dim, d = shape.kv_heads, shape.head_dim, shape.dtype_bytes
    staging = 2 * k * active * dim * d
    if profile.method == "quest":
        encode = k * active * dim * d + k * dim * 2 * 4
        compares = 2 * k * dim * max(0, active - 1)
        return Cost(bytes=(staging + encode) * sparse_layers,
                    comparisons=compares * sparse_layers,
                    conversions=k * active * dim * sparse_layers)
    if profile.method == "fier":
        encode = 2 * k * active * dim * d + k * dim * (4 + 2 * d)
        comparisons = k * dim * (2 * max(0, active - 1) + active)
        return Cost(flops=2 * k * dim * sparse_layers, bytes=(staging + encode) * sparse_layers,
                    comparisons=comparisons * sparse_layers,
                    bitwise_ops=k * dim * active * 2 * sparse_layers,
                    conversions=k * dim * (3 * active + 2) * sparse_layers)
    words, reps = math.ceil(dim / 32), math.ceil(dim / profile.dim_group)
    encoded = k * active * (words * 2 * 4 + reps * 2 * 2)
    encode_reads = 2 * k * active * dim * d
    flops = k * active * (2 * dim + 2 * reps)
    comparisons = k * dim * (2 * max(0, active - 1) + 2 * active)
    return Cost(flops=flops * sparse_layers, bytes=(staging + encode_reads + encoded) * sparse_layers,
                comparisons=comparisons * sparse_layers,
                bitwise_ops=2 * k * active * dim * sparse_layers,
                conversions=2 * k * active * dim * sparse_layers)


def _prepack(profile: Profile, shape: ModelShape, tokens: int, sparse_layers: int) -> Cost:
    if profile.method == "full":
        return Cost()
    k, dim, d = shape.kv_heads, shape.head_dim, shape.dtype_bytes
    elements = k * tokens * dim
    metadata = _metadata_bytes_per_layer(profile, shape, tokens)
    if profile.method == "quest":
        traffic = elements * (d + 4 + 2 * 4) + metadata
        groups = math.ceil(tokens / profile.page_size)
        comparisons = 2 * k * dim * (tokens - groups)
        return Cost(bytes=traffic * sparse_layers, comparisons=comparisons * sparse_layers,
                    conversions=elements * sparse_layers)
    if profile.method == "fier":
        groups = math.ceil(tokens / profile.token_group)
        flops = 2 * elements + 2 * k * groups * dim
        traffic = elements * (2 * d + 2 * d) + metadata
        comparisons = k * dim * (2 * (tokens - groups) + tokens)
        words = math.ceil(tokens / 32)
        if profile.fier_kernel == "fier":
            # The public wrapper converts max to scale in place, then makes the
            # packed [KVH,D,word] result word-major and contiguous.
            metadata_elements = k * groups * dim
            packed_bytes = k * dim * words * 4
            flops += metadata_elements
            traffic += metadata_elements * (3 * d) + 2 * packed_bytes
        return Cost(flops * sparse_layers, 0, traffic * sparse_layers,
                    comparisons=comparisons * sparse_layers,
                    integer_ops=k * dim * words * 31 * sparse_layers,
                    bitwise_ops=2 * elements * sparse_layers,
                    conversions=(4 * elements + 2 * k * groups * dim) * sparse_layers)
    reps = math.ceil(dim / profile.dim_group)
    flops = elements * 2 + k * tokens * reps * 2
    traffic = 2 * elements * d + metadata
    groups = math.ceil(tokens / profile.token_group)
    comparisons = k * dim * (2 * (tokens - groups) + 2 * tokens)
    return Cost(flops * sparse_layers, 0, traffic * sparse_layers,
                comparisons=comparisons * sparse_layers,
                bitwise_ops=2 * elements * sparse_layers,
                conversions=2 * elements * sparse_layers)


def estimate(profile: Profile, shape: ModelShape, context: int, generated: int, budget: int,
             full_layers: int) -> tuple[dict[str, Cost], dict[str, float | int | str]]:
    sparse_layers = shape.layers - full_layers
    stages: dict[str, Cost] = {name: Cost() for name in (
        "prepack", "frontend", "qkv_rope", "retrieval_score", "topk_selection", "gather",
        "attention", "output_projection", "mlp", "final_norm_lm_head", "token_selection",
        "kv_update", "retrieval_update",
    )}
    stages["prepack"] = _prepack(profile, shape, context - 1, sparse_layers)
    common = _common_stages(shape)
    for step in range(generated):
        prefix = context - 1 + step
        tokens = prefix + 1
        for name, cost in common.items():
            stages[name] = stages[name] + cost
        if profile.method == "full":
            stages["gather"] = stages["gather"] + _full_gather(shape, tokens) * shape.layers
            stages["attention"] = stages["attention"] + _attention(shape, tokens) * shape.layers
        else:
            stages["gather"] = stages["gather"] + _full_gather(shape, tokens) * full_layers
            stages["attention"] = stages["attention"] + _attention(shape, tokens) * full_layers
            stages["retrieval_score"] = stages["retrieval_score"] + _score(profile, shape, prefix) * sparse_layers
            stages["topk_selection"] = stages["topk_selection"] + _topk(profile, shape, prefix, budget) * sparse_layers
            selected = math.ceil(budget / profile.page_size) * profile.page_size if profile.method == "quest" else budget
            stages["attention"] = stages["attention"] + _attention(
                shape, selected, valid_mask=profile.method == "quest", indexed=True,
                split_tokens=256) * sparse_layers
        stages["token_selection"] = stages["token_selection"] + Cost(
            bytes=shape.vocab * 4, comparisons=shape.vocab - 1)
        stages["kv_update"] = stages["kv_update"] + _kv_update(shape)
        group = profile.page_size if profile.method == "quest" else profile.token_group
        active = (prefix % group) + 1 if group else 0
        stages["retrieval_update"] = stages["retrieval_update"] + _retrieval_update(
            profile, shape, active, sparse_layers)

    capacity_tokens = context + generated
    base_k_per_layer = shape.kv_heads * capacity_tokens * shape.head_dim * shape.dtype_bytes
    base_kv = 2 * base_k_per_layer * shape.layers
    metadata = _metadata_bytes_per_layer(profile, shape, capacity_tokens) * sparse_layers
    total = Cost()
    online = Cost()
    for name, cost in stages.items():
        total += cost
        if name != "prepack":
            online += cost
    summary: dict[str, float | int | str] = {
        "method": profile.label,
        "online_flops": online.flops,
        "prepack_flops": stages["prepack"].flops,
        "total_flops": total.flops,
        "popcount_ops": total.popcount,
        "comparison_ops": total.comparisons,
        "integer_ops": total.integer_ops,
        "bitwise_ops": total.bitwise_ops,
        "conversion_ops": total.conversions,
        "exp_ops": total.exp_ops,
        "rsqrt_ops": total.rsqrt_ops,
        "topk_input_elements": total.topk_inputs,
        "logical_bytes": total.bytes,
        "metadata_bytes": metadata,
        "base_kv_bytes": base_kv,
        "total_cache_bytes": base_kv + metadata,
        "total_cache_vs_full_kv": (base_kv + metadata) / base_kv,
    }
    return stages, summary


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader(); writer.writerows(materialized)


def _markdown(summary: list[dict[str, float | int | str]], generated: int,
              blocks: int) -> str:
    lines = [
        "| Method | TFLOP | Compare (G) | Integer (G) | Bitwise (G) | POpcount (G) | "
        "Convert (G) | Exp (M) | rsqrt (M) | Top-k inputs (G) | Logical traffic (TB) | "
        "Metadata (MiB) | Total cache / Full KV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary:
        lines.append(
            f"| {x['method']} | {int(x['total_flops']) / 1e12:.3f} | "
            f"{int(x['comparison_ops']) / 1e9:.3f} | {int(x['integer_ops']) / 1e9:.3f} | "
            f"{int(x['bitwise_ops']) / 1e9:.3f} | {int(x['popcount_ops']) / 1e9:.3f} | "
            f"{int(x['conversion_ops']) / 1e9:.3f} | {int(x['exp_ops']) / 1e6:.3f} | "
            f"{int(x['rsqrt_ops']) / 1e6:.3f} | {int(x['topk_input_elements']) / 1e9:.3f} | "
            f"{int(x['logical_bytes']) / 1e12:.3f} | "
            f"{int(x['metadata_bytes']) / 2**20:.2f} | "
            f"{float(x['total_cache_vs_full_kv']):.3f}x |"
        )
    return "\n".join([
        f"# Resource cost summary ({blocks} blocks, {generated} generated tokens/block)", "",
        "Values are means per block. Analytical counts are identical for equal shapes.", "", *lines, "",
        "FMA=2 FLOPs. Comparisons, integer arithmetic, bitwise operations, 32-bit CUDA popcount, "
        "conversions, exp, and rsqrt remain separate. topk_input_elements reports the exact "
        "selection workload cardinality because the internal CUDA top-k comparator/integer "
        "instruction count is backend- and data-dependent.", "",
        "Logical traffic is operator-level tensor read+write traffic.", "",
        "All sparse methods retain the BF16 K/V cache. Metadata is therefore additional storage; "
        "the final column reports total cache storage relative to Full K/V.", "",
    ])


def _stage_markdown(rows: list[dict[str, object]], generated: int) -> str:
    lines = [
        "| Method | Stage | GFLOP/token | Compare G/token | Integer G/token | Bitwise G/token | "
        "POpcount G/token | Convert G/token | Exp M/token | rsqrt M/token | "
        "Top-k inputs M/token | Logical GB/token |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['stage']} | {float(row['flops_per_token']) / 1e9:.6f} | "
            f"{float(row['comparisons_per_token']) / 1e9:.6f} | "
            f"{float(row['integer_ops_per_token']) / 1e9:.6f} | "
            f"{float(row['bitwise_ops_per_token']) / 1e9:.6f} | "
            f"{float(row['popcount_per_token']) / 1e9:.6f} | "
            f"{float(row['conversions_per_token']) / 1e9:.6f} | "
            f"{float(row['exp_ops_per_token']) / 1e6:.6f} | "
            f"{float(row['rsqrt_ops_per_token']) / 1e6:.6f} | "
            f"{float(row['topk_inputs_per_token']) / 1e6:.6f} | "
            f"{float(row['bytes_per_token']) / 1e9:.6f} |"
        )
    return "\n".join([
        "# Stage-level resource costs", "",
        "Values are per-generated-token means; the one-time prepack row is divided by the "
        f"{generated}-token block size.", "", *lines, "",
    ])

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLOP/POpcount/traffic/cache accounting")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--generate-tokens", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--full-layers", type=int, default=2)
    parser.add_argument("--output-dir", default="results/resource_cost_32k_b4096_t256_blocks10")
    return parser.parse_args()


def main() -> None:
    args = arguments(); shape = ModelShape(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    detail, summary = [], []
    for profile in PROFILES:
        stages, row = estimate(profile, shape, args.context_length, args.generate_tokens,
                               args.budget, args.full_layers)
        summary.append(row)
        for stage, cost in stages.items():
            item = {"method": profile.label, "stage": stage, **asdict(cost)}
            for name in cost.__dataclass_fields__:
                item[name + "_per_token"] = getattr(cost, name) / args.generate_tokens
            detail.append(item)
    _write_csv(output / "detail.csv", detail)
    _write_csv(output / "summary.csv", summary)
    (output / "summary.md").write_text(
        _markdown(summary, args.generate_tokens, args.num_blocks))
    (output / "stages.md").write_text(_stage_markdown(detail, args.generate_tokens))
    (output / "metadata.json").write_text(json.dumps({
        "context_length": args.context_length, "generate_tokens": args.generate_tokens,
        "num_blocks": args.num_blocks, "budget": args.budget, "full_layers": args.full_layers,
        "model": asdict(shape), "profiles": [asdict(x) for x in PROFILES],
        "counting_contract": {
            "fma_flops": 2,
            "separate_operation_classes": [
                "flops", "comparisons", "integer_ops", "bitwise_ops", "popcount",
                "conversions", "exp_ops", "rsqrt_ops", "topk_inputs"
            ],
            "logical_bytes": "operator-level tensor reads plus writes",
            "topk_inputs": "input cardinality, not an invented comparator count",
        },
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
