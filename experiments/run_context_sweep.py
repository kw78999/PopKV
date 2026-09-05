from __future__ import annotations

"""Resumable PG19 context sweep with per-case process isolation and ETA."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_METHODS = "full,quest,fier-author,popkv-q,popkv-s"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated PG19 context-length cases")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--context-start-k", type=int, default=32)
    parser.add_argument("--context-end-k", type=int, default=128)
    parser.add_argument("--context-step-k", type=int, default=1)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--generate-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="outputs/context_sweep_c32k_128k_step1k_b4096_blocks4_t256",
    )
    parser.add_argument("--text-file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def expected_methods(value: str) -> list[str]:
    methods = list(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def completed_case(case_dir: Path, methods: list[str], expected_tokens: int) -> bool:
    summary = case_dir / "summary.csv"
    if not summary.exists():
        return False
    try:
        rows = list(csv.DictReader(summary.open()))
        requested = [row for row in rows if row["method"] in set(methods)]
        return (
            {row["method"] for row in requested} == set(methods)
            and all(int(row["tokens"]) == expected_tokens for row in requested)
            and all("retrieval_search_ms_per_token" in row for row in requested)
        )
    except (KeyError, OSError, ValueError):
        return False


def aggregate(root: Path, contexts: list[int]) -> None:
    combined: list[dict[str, str]] = []
    for context in contexts:
        case_dir = root / f"c{context}"
        summary = case_dir / "summary.csv"
        if not summary.exists():
            continue
        for row in csv.DictReader(summary.open()):
            combined.append(
                {"context_length": str(context), "case_dir": str(case_dir), **row}
            )
    if not combined:
        return
    output = root / "context_sweep_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)


def main() -> None:
    args = arguments()
    if args.context_step_k <= 0 or args.context_end_k < args.context_start_k:
        raise ValueError("invalid context range")
    contexts = [
        value * 1024
        for value in range(args.context_start_k, args.context_end_k + 1, args.context_step_k)
    ]
    methods = expected_methods(args.methods)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    expected_tokens = args.num_blocks * args.generate_tokens
    pending = [
        context for context in contexts
        if args.rerun_complete or not completed_case(root / f"c{context}", methods, expected_tokens)
    ]
    print(
        f"[sweep] total={len(contexts)} complete={len(contexts) - len(pending)} "
        f"pending={len(pending)} range={contexts[0]}..{contexts[-1]} step={args.context_step_k * 1024}",
        flush=True,
    )
    durations: list[float] = []
    sweep_started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for pending_index, context in enumerate(pending, 1):
        case_dir = root / f"c{context}"
        command = [
            sys.executable, "-m", "experiments.evaluate_pg19",
            "--model", args.model,
            "--methods", ",".join(methods),
            "--context-length", str(context),
            "--budget", str(args.budget),
            "--num-blocks", str(args.num_blocks),
            "--generate-tokens", str(args.generate_tokens),
            "--prefill-chunk-size", str(args.prefill_chunk_size),
            "--seed", str(args.seed),
            "--output-dir", str(case_dir),
        ]
        if args.text_file:
            command.extend(["--text-file", args.text_file])
        print(
            f"[sweep] start={pending_index}/{len(pending)} context={context} "
            f"command={' '.join(command)}",
            flush=True,
        )
        if args.dry_run:
            continue
        case_started = time.perf_counter()
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env)
        duration = time.perf_counter() - case_started
        if result.returncode != 0:
            print(f"[sweep] FAILED context={context} exit={result.returncode}", flush=True)
            raise SystemExit(result.returncode)
        if not completed_case(case_dir, methods, expected_tokens):
            raise RuntimeError(f"context {context} exited successfully but results are incomplete")
        durations.append(duration)
        aggregate(root, contexts)
        remaining_cases = len(pending) - pending_index
        remaining_seconds = sum(durations) / len(durations) * remaining_cases
        eta = datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)
        elapsed = time.perf_counter() - sweep_started
        print(
            f"[sweep] done={pending_index}/{len(pending)} "
            f"overall={len(contexts) - len(pending) + pending_index}/{len(contexts)} "
            f"context={context} case_h={duration / 3600:.2f} elapsed_h={elapsed / 3600:.2f} "
            f"eta_utc={eta:%Y-%m-%dT%H:%M:%SZ}",
            flush=True,
        )

    aggregate(root, contexts)
    manifest = {
        "model": args.model,
        "methods": methods,
        "contexts": contexts,
        "budget": args.budget,
        "num_blocks": args.num_blocks,
        "generate_tokens": args.generate_tokens,
        "prefill_chunk_size": args.prefill_chunk_size,
        "seed": args.seed,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not args.dry_run:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[sweep] results={root}", flush=True)


if __name__ == "__main__":
    main()

