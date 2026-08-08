from __future__ import annotations

"""CUDA micro-evaluation for PopKV encoding overhead.

Latency mode intentionally benchmarks only existing production pack/score calls
plus a benchmark-only query encoder.  It does not run model inference.
"""

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from popkv import reference_query_encode_g32


OPERATIONS = ("query_encode", "production_score", "key_prefix_encode", "key_incremental_encode")
DEFAULT_CONTEXTS = (4096, 16384, 32768, 65536)
OCCUPANCIES = tuple(range(1, 33))


@dataclass(frozen=True)
class BenchmarkCase:
    operation: str
    context_length: int | None
    active_group_occupancy: int | None
    scope: str
    calls_per_operation: int


def parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PopKV query/key encoding CUDA micro-evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--contexts", type=parse_ints, default=DEFAULT_CONTEXTS)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dim-group-size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs/encoding_overhead")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    positive = ("warmup", "repetitions", "trials", "q_heads", "kv_heads", "head_dim", "layers")
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("warmup, repetitions, trials, heads, head-dim, and layers must be positive")
    if args.layers < 3:
        parser.error("--layers must be at least 3 to report sparse layers after two full layers")
    if args.q_heads < args.kv_heads or args.q_heads % args.kv_heads:
        parser.error("--q-heads must be a multiple of --kv-heads")
    return args


def benchmark_cases(contexts: Iterable[int], layers: int) -> list[BenchmarkCase]:
    sparse_layers = layers - 2
    cases = [
        BenchmarkCase("query_encode", None, None, "one_layer", 1),
        BenchmarkCase("query_encode", None, None, f"{sparse_layers}_sparse_layers", sparse_layers),
    ]
    for context in contexts:
        cases.extend((
            BenchmarkCase("production_score", context, None, "one_layer", 1),
            BenchmarkCase("production_score", context, None, f"{sparse_layers}_sparse_layers", sparse_layers),
            BenchmarkCase("key_prefix_encode", context, None, "one_layer", 1),
            BenchmarkCase("key_prefix_encode", context, None, f"{layers}_layers", layers),
        ))
    for occupancy in OCCUPANCIES:
        cases.extend((
            BenchmarkCase("key_incremental_encode", None, occupancy, "one_layer", 1),
            BenchmarkCase("key_incremental_encode", None, occupancy, f"{layers}_layers", layers),
        ))
    return cases


def expected_shapes(args: argparse.Namespace) -> dict[str, object]:
    words = math.ceil(args.head_dim / 32)
    return {
        "query": [1, args.q_heads, args.head_dim],
        "query_bitplane_each": [1, args.q_heads, words],
        "query_representative_each": [1, args.q_heads, math.ceil(args.head_dim / args.dim_group_size)],
        "key_prefix_by_context": {str(t): [args.kv_heads, t, args.head_dim] for t in args.contexts},
        "packed_key_by_context": {str(t): [args.kv_heads, t, words] for t in args.contexts},
        "key_representative_by_context": {
            str(t): [args.kv_heads, t, math.ceil(args.head_dim / args.dim_group_size)]
            for t in args.contexts
        },
        "occupancies": list(OCCUPANCIES),
    }


def validate_reference() -> None:
    for dim in (32, 64, 96, 127, 128):
        q = torch.linspace(-3, 3, 2 * 3 * dim).view(2, 3, dim)
        sign, mag, low, delta = reference_query_encode_g32(q)
        shape = (2, 3, math.ceil(dim / 32))
        assert sign.shape == mag.shape == low.shape == delta.shape == shape
        assert sign.dtype == mag.dtype == torch.int32
        assert low.dtype == delta.dtype == torch.float32
        assert torch.isfinite(low).all() and torch.isfinite(delta).all()


def _packed_storage(
    kv_heads: int, capacity: int, head_dim: int, device: torch.device, dim_group_size: int = 32,
):
    words = math.ceil(head_dim / 32)
    reps = math.ceil(head_dim / dim_group_size)
    bit_shape = (kv_heads, capacity, words)
    rep_shape = (kv_heads, capacity, reps)
    return (
        torch.empty(bit_shape, device=device, dtype=torch.int32),
        torch.empty(bit_shape, device=device, dtype=torch.int32),
        torch.empty(rep_shape, device=device, dtype=torch.float16),
        torch.empty(rep_shape, device=device, dtype=torch.float16),
    )


def measure_cuda(operation: Callable[[], object], args: argparse.Namespace) -> list[float]:
    operation()  # JIT/dispatch initialization is outside all timing boundaries.
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()
    results = []
    for _ in range(args.trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repetitions):
            operation()
        end.record()
        torch.cuda.synchronize()
        results.append(float(start.elapsed_time(end)) * 1000.0 / args.repetitions)
    return results


def repeat_calls(operation: Callable[[], object], count: int) -> None:
    for _ in range(count):
        operation()


def raw_rows(case: BenchmarkCase, values: list[float], args: argparse.Namespace) -> list[dict[str, object]]:
    return [{
        "operation": case.operation,
        "metric": "latency_us",
        "context_length": case.context_length,
        "active_group_occupancy": case.active_group_occupancy,
        "scope": case.scope,
        "trial": trial,
        "latency_us": value,
        "calls_per_operation": case.calls_per_operation,
        "repetitions": args.repetitions,
    } for trial, value in enumerate(values, 1)]


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[x] for x in ("operation", "metric", "context_length", "active_group_occupancy", "scope", "calls_per_operation"))
        groups.setdefault(key, []).append(float(row["latency_us"]))
    output = []
    for key, values in groups.items():
        operation, metric, context, occupancy, scope, calls = key
        output.append({
            "operation": operation, "metric": metric, "context_length": context,
            "active_group_occupancy": occupancy, "scope": scope,
            "median_us": statistics.median(values), "mean_us": statistics.mean(values),
            "std_us": statistics.pstdev(values), "min_us": min(values), "max_us": max(values),
            "calls_per_operation": calls, "repetitions": args.repetitions, "trials": args.trials,
            "us_per_token": (statistics.median(values) / int(context)
                             if operation == "key_prefix_encode" and context and scope == "one_layer" else ""),
            "isolated_query_encode_to_score_ratio": "",
        })

    for scope in ("one_layer", f"{args.layers - 2}_sparse_layers"):
        query = next(x for x in output if x["operation"] == "query_encode" and x["scope"] == scope)
        for row in output:
            if row["operation"] == "production_score" and row["scope"] == scope:
                row["isolated_query_encode_to_score_ratio"] = float(query["median_us"]) / float(row["median_us"])
    return output


def add_cycle_means(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    for scope in ("one_layer", f"{args.layers}_layers"):
        selected = [r for r in rows if r["operation"] == "key_incremental_encode" and r["scope"] == scope]
        for trial in range(1, args.trials + 1):
            values = [float(r["latency_us"]) for r in selected if r["trial"] == trial]
            assert len(values) == 32
            case = BenchmarkCase("key_incremental_encode", None, None, scope, 1 if scope == "one_layer" else args.layers)
            row = raw_rows(case, [statistics.mean(values)], args)[0]
            row["trial"] = trial
            row["metric"] = "average_cycle_incremental_pack_us"
            rows.append(row)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, object]], args: argparse.Namespace) -> None:
    print("\nContext comparison (primary statistic: median)")
    print("context  query_encode_us  production_score_us  isolated_query_encode_to_score_ratio")
    query = next(x for x in summary if x["operation"] == "query_encode" and x["scope"] == "one_layer")
    for context in args.contexts:
        score = next(x for x in summary if x["operation"] == "production_score" and x["scope"] == "one_layer" and x["context_length"] == context)
        print(f"{context:7d}  {float(query['median_us']):15.3f}  {float(score['median_us']):19.3f}  {float(score['isolated_query_encode_to_score_ratio']):38.6f}")
    print("\nIncremental key pack medians (us)")
    print("occupancy  one_layer  synthetic_layers")
    for occupancy in (1, 8, 16, 24, 32):
        one = next(x for x in summary if x["operation"] == "key_incremental_encode" and x["metric"] == "latency_us" and x["scope"] == "one_layer" and x["active_group_occupancy"] == occupancy)
        many = next(x for x in summary if x["operation"] == "key_incremental_encode" and x["metric"] == "latency_us" and x["scope"] == f"{args.layers}_layers" and x["active_group_occupancy"] == occupancy)
        print(f"{occupancy:9d}  {float(one['median_us']):9.3f}  {float(many['median_us']):16.3f}")


def run_cuda(args: argparse.Namespace) -> None:
    from popkv_cuda import allocate_query_encode_outputs, pack_keys_into, query_encode, score

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    query = torch.randn((1, args.q_heads, args.head_dim), device=device, dtype=dtype)
    query_out = allocate_query_encode_outputs(query, dim_group_size=args.dim_group_size)
    head_to_kv = (torch.arange(args.q_heads, device=device, dtype=torch.long) * args.kv_heads // args.q_heads).contiguous()
    rows: list[dict[str, object]] = []
    cases = benchmark_cases(args.contexts, args.layers)

    query_op = lambda: query_encode(query, out=query_out, dim_group_size=args.dim_group_size)
    for case in (x for x in cases if x.operation == "query_encode"):
        op = query_op if case.calls_per_operation == 1 else lambda c=case.calls_per_operation: repeat_calls(query_op, c)
        rows.extend(raw_rows(case, measure_cuda(op, args), args))

    for context in args.contexts:
        keys = torch.randn((args.kv_heads, context, args.head_dim), device=device, dtype=dtype)
        packed = _packed_storage(args.kv_heads, context, args.head_dim, device, args.dim_group_size)
        pack_op = lambda k=keys, p=packed: pack_keys_into(
            k, p, token_offset=0, dim_group_size=args.dim_group_size,
        )
        pack_op(); torch.cuda.synchronize()
        for case in (x for x in cases if x.context_length == context and x.operation == "key_prefix_encode"):
            op = pack_op if case.calls_per_operation == 1 else lambda c=case.calls_per_operation: repeat_calls(pack_op, c)
            rows.extend(raw_rows(case, measure_cuda(op, args), args))
        for case in (x for x in cases if x.context_length == context and x.operation == "production_score"):
            score_op = lambda p=packed, t=context: score(
                query, p, head_to_kv=head_to_kv, tokens=t, dim_group_size=args.dim_group_size,
            )
            op = score_op if case.calls_per_operation == 1 else lambda c=case.calls_per_operation: repeat_calls(score_op, c)
            rows.extend(raw_rows(case, measure_cuda(op, args), args))
        del keys, packed

    incremental_inputs = {m: torch.randn((args.kv_heads, m, args.head_dim), device=device, dtype=dtype) for m in OCCUPANCIES}
    incremental_out = _packed_storage(args.kv_heads, 32, args.head_dim, device, args.dim_group_size)
    for occupancy in OCCUPANCIES:
        pack_op = lambda k=incremental_inputs[occupancy]: pack_keys_into(
            k, incremental_out, token_offset=0, dim_group_size=args.dim_group_size,
        )
        for case in (x for x in cases if x.operation == "key_incremental_encode" and x.active_group_occupancy == occupancy):
            op = pack_op if case.calls_per_operation == 1 else lambda c=case.calls_per_operation: repeat_calls(pack_op, c)
            rows.extend(raw_rows(case, measure_cuda(op, args), args))

    add_cycle_means(rows, args)
    summary = summarize(rows, args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "encoding_overhead_raw.csv", rows)
    write_csv(output / "encoding_overhead_summary.csv", summary)
    (output / "encoding_overhead_config.json").write_text(json.dumps(vars(args), indent=2, default=list) + "\n")
    print_summary(summary, args)
    print(f"\nresults={output}")


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    validate_reference()
    plan = {
        "config": vars(args), "expected_shapes": expected_shapes(args),
        "case_count": len(benchmark_cases(args.contexts, args.layers)),
        "contexts": list(args.contexts), "occupancies": list(OCCUPANCIES),
        "operations": list(OPERATIONS),
    }
    if args.dry_run or args.validate_only:
        print(json.dumps(plan, indent=2, default=list))
        print("reference_validation=ok")
        return 0
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        print("CUDA is required for latency measurement.")
        print("Use --dry-run / --validate-only locally, then run the following command on a CUDA GPU.")
        print("python benchmark_encoding_overhead.py --device cuda --dtype bfloat16 --contexts 4096,16384,32768,65536 --q-heads 32 --kv-heads 8 --head-dim 128 --dim-group-size 32 --layers 32 --warmup 50 --repetitions 100 --trials 10 --output-dir outputs/encoding_overhead")
        return 2
    run_cuda(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
