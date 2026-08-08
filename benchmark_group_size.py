from __future__ import annotations

"""Lightweight CUDA benchmark for PopKV G_dim score latency."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(x) for x in value.split(",") if x.strip())
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PopKV dimension-group score microbenchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--contexts", type=parse_ints, default=(4096, 16384, 32768, 65536))
    parser.add_argument("--dim-group-sizes", type=parse_ints, default=(16, 32, 64))
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/group_size_score")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if set(args.dim_group_sizes) - {16, 32, 64}:
        parser.error("--dim-group-sizes supports only 16,32,64")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be a multiple of --kv-heads")
    if min(args.q_heads, args.kv_heads, args.head_dim, args.warmup, args.repetitions, args.trials) <= 0:
        parser.error("heads, dimensions, warmup, repetitions, and trials must be positive")
    return args


def cases(args: argparse.Namespace) -> list[dict[str, int]]:
    return [{"dim_group_size": group, "context_length": context}
            for group in args.dim_group_sizes for context in args.contexts]


def measure(operation, args: argparse.Namespace) -> list[float]:
    operation(); torch.cuda.synchronize()
    for _ in range(args.warmup): operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(args.trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repetitions): operation()
        end.record(); torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)) / args.repetitions)
    return values


def run(args: argparse.Namespace) -> None:
    from popkv_cuda import pack_keys, score
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    query = torch.randn(1, args.q_heads, args.head_dim, device=device, dtype=dtype)
    mapping = (torch.arange(args.q_heads, device=device, dtype=torch.long) * args.kv_heads // args.q_heads).contiguous()
    raw, summary = [], []
    for case in cases(args):
        group, context = case["dim_group_size"], case["context_length"]
        keys = torch.randn(args.kv_heads, context, args.head_dim, device=device, dtype=dtype)
        packed = pack_keys(keys.contiguous(), dim_group_size=group)
        torch.cuda.synchronize()
        operation = lambda p=packed, g=group, t=context: score(
            query, p, head_to_kv=mapping, tokens=t, dim_group_size=g,
        )
        values = measure(operation, args)
        raw.extend({**case, "trial": trial, "score_ms": value,
                    "repetitions": args.repetitions} for trial, value in enumerate(values, 1))
        summary.append({**case, "median_score_ms": statistics.median(values),
                        "mean_score_ms": statistics.mean(values),
                        "std_score_ms": statistics.pstdev(values), "min_score_ms": min(values),
                        "max_score_ms": max(values), "repetitions": args.repetitions, "trials": args.trials})
        del keys, packed
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("group_size_score_raw.csv", raw), ("group_size_score_summary.csv", summary)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print("G_dim  context  median_score_ms")
    for row in summary:
        print(f"{row['dim_group_size']:5d}  {row['context_length']:7d}  {row['median_score_ms']:.6f}")
    print(f"results={output}")


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    plan = {"config": vars(args), "cases": cases(args),
            "fixed_token_group_size": 32, "fixed_packing_width": 32}
    if args.dry_run or args.validate_only:
        print(json.dumps(plan, indent=2, default=list)); return 0
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        print("CUDA is required for latency measurement. Use --dry-run / --validate-only locally.")
        return 2
    run(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
