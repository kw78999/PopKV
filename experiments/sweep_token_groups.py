from __future__ import annotations

"""Sweep Pop-KV token-group size while reusing one model and one PG19 block."""

import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from popkv import LlamaSparseDecoder, RetrievalConfig, load_hf_model
from experiments.evaluate_pg19 import load_blocks, sync


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pop-KV tok-G sweep (PPL and retrieval-update latency)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--generate-tokens", type=int, default=500)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--tok-g-start", type=int, default=2)
    parser.add_argument("--tok-g-stop", type=int, default=500)
    parser.add_argument("--tok-g-step", type=int, default=2)
    parser.add_argument("--dim-group-size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--dim-kernel", choices=("auto", "g16", "g32", "g64-u32", "g64-u64"), default="auto")
    parser.add_argument("--full-layers", default="0,1")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pg19-cache-dir", default="data/pg19")
    parser.add_argument("--pg19-max-books", type=int, default=64)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip tok-G values already present in summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.tok_g_start <= 0 or args.tok_g_stop < args.tok_g_start or args.tok_g_step <= 0:
        parser.error("tok-G requires 0 < start <= stop and step > 0")
    if args.generate_tokens <= 0 or args.num_blocks <= 0 or args.context_length <= 1 or args.budget <= 1:
        parser.error("generate-tokens, num-blocks, context-length, and budget must be positive")
    if args.dim_group_size != 32 and any(
        value not in (8, 16, 32, 64, 128) for value in token_groups(args)
    ):
        parser.error("arbitrary tok-G currently requires --dim-group-size 32")
    return args


def token_groups(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(range(args.tok_g_start, args.tok_g_stop + 1, args.tok_g_step))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_completed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_int = {"case", "tok_g", "tokens"}
    numeric_float = {
        "perplexity", "mean_nll", "key_metadata_encoding_ms_total",
        "key_metadata_encoding_ms_per_token", "retrieval_group_sealing_ms_total",
        "retrieval_group_sealing_ms_per_token", "prepack_ms", "case_elapsed_seconds",
    }
    for row in rows:
        for key in numeric_int:
            row[key] = int(row[key])
        for key in numeric_float:
            row[key] = float(row[key])
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def run_case(
    model: Any,
    prepared_blocks: list[tuple[Any, torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    tok_g: int,
) -> dict[str, float]:
    config = RetrievalConfig(
        budget=args.budget,
        group_size=tok_g,
        dim_group_size=args.dim_group_size,
        dim_kernel=args.dim_kernel,
        full_layers=tuple(int(x) for x in args.full_layers.split(",") if x.strip()),
    )
    config.validate()
    decoder = LlamaSparseDecoder(model, config)
    nll_total = 0.0
    metadata_ms = 0.0
    sealing_ms = 0.0
    prepack_ms = 0.0
    tokens = 0

    for prefix, current_ids, targets in prepared_blocks:
        pack_started = time.perf_counter()
        retrieval = decoder.build_retrieval_caches(
            prefix, method="popkv", reserve_tokens=args.generate_tokens + 1,
        )
        sync(decoder.device)
        prepack_ms += (time.perf_counter() - pack_started) * 1000.0
        kv = decoder.build_static_kv_cache(prefix, reserve_tokens=args.generate_tokens + 1)

        # Compile/JIT and warm up the decode path without mutating either cache.
        decoder.decode(kv, current_ids[0], method="popkv", retrieval_caches=retrieval)
        sync(decoder.device)

        for current, target in zip(current_ids, targets):
            logits, _, new_k, new_v = decoder.decode(
                kv, current, method="popkv", retrieval_caches=retrieval, return_new_kv=True,
            )
            decoder.append_kv(kv, new_k, new_v)
            retrieval_timer = decoder.append_retrieval_caches(retrieval, new_k, measure=True)
            sync(decoder.device)
            parts = retrieval_timer.totals()
            metadata_ms += parts["k_metadata_encoding"]
            sealing_ms += parts["group_sealing"]
            nll_total += float(F.cross_entropy(logits[None], target.to(decoder.device)[None]).item())
        tokens += len(targets)
        del retrieval, kv

    mean_nll = nll_total / tokens
    del decoder
    return {
        "tokens": tokens,
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "key_metadata_encoding_ms_total": metadata_ms,
        "key_metadata_encoding_ms_per_token": metadata_ms / tokens,
        "retrieval_group_sealing_ms_total": sealing_ms,
        "retrieval_group_sealing_ms_per_token": sealing_ms / tokens,
        "prepack_ms": prepack_ms,
    }


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    groups = token_groups(args)
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or f"outputs/tok_g_sweep_{stamp}")
    plan = {
        "model": args.model,
        "tok_g_values": list(groups),
        "total_cases": len(groups),
        "blocks_per_case": args.num_blocks,
        "tokens_per_block": args.generate_tokens,
        "tokens_per_case": args.num_blocks * args.generate_tokens,
        "context_length": args.context_length,
        "budget": args.budget,
        "dim_group_size": args.dim_group_size,
        "output_dir": str(output),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; use --dry-run to validate the sweep plan")

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"{output} already exists; choose another --output-dir or pass --resume")
    config_path.write_text(json.dumps(vars(args) | plan | {"started_at_utc": iso_utc(utc_now())}, indent=2) + "\n")
    summary_path = output / "summary.csv"
    rows = read_completed(summary_path) if args.resume else []
    completed_groups = {int(row["tok_g"]) for row in rows}

    torch.manual_seed(args.seed)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attention=args.attention,
        trust_remote_code=args.trust_remote_code,
    )
    blocks = load_blocks(tokenizer, args)
    base_config = RetrievalConfig(
        budget=args.budget,
        group_size=groups[0],
        dim_group_size=args.dim_group_size,
        dim_kernel=args.dim_kernel,
        full_layers=tuple(int(x) for x in args.full_layers.split(",") if x.strip()),
    )
    base_config.validate()
    base_decoder = LlamaSparseDecoder(model, base_config)
    prepared_blocks = []
    for block in blocks:
        prefix_ids = block[:args.context_length - 1]
        current_ids = block[
            args.context_length - 1:args.context_length - 1 + args.generate_tokens
        ]
        targets = block[args.context_length:args.context_length + args.generate_tokens]
        prefix = base_decoder.build_prefix_cache(prefix_ids)
        sync(base_decoder.device)
        prepared_blocks.append((prefix, current_ids, targets))
    del base_decoder

    pending = [value for value in groups if value not in completed_groups]
    run_started = time.perf_counter()
    print(
        f"sweep_start total={len(groups)} completed={len(rows)} pending={len(pending)} "
        f"started_utc={iso_utc(utc_now())}",
        flush=True,
    )
    for tok_g in pending:
        case_started_wall = utc_now()
        case_started = time.perf_counter()
        metrics = run_case(model, prepared_blocks, args, tok_g)
        elapsed = time.perf_counter() - case_started
        row = {
            "case": groups.index(tok_g) + 1,
            "tok_g": tok_g,
            **metrics,
            "case_elapsed_seconds": elapsed,
            "started_at_utc": iso_utc(case_started_wall),
            "completed_at_utc": iso_utc(utc_now()),
        }
        rows.append(row)
        rows.sort(key=lambda item: int(item["case"]))
        write_summary(summary_path, rows)

        completed = len(rows)
        durations = [float(item["case_elapsed_seconds"]) for item in rows]
        average = sum(durations) / len(durations)
        remaining_seconds = average * (len(groups) - completed)
        eta = utc_now() + timedelta(seconds=remaining_seconds)
        print(
            f"progress={completed}/{len(groups)} ({100.0 * completed / len(groups):.1f}%) "
            f"tok_g={tok_g} ppl={metrics['perplexity']:.6f} "
            f"key_metadata_encoding_ms/token={metrics['key_metadata_encoding_ms_per_token']:.6f} "
            f"retrieval_group_sealing_ms/token={metrics['retrieval_group_sealing_ms_per_token']:.6f} "
            f"case={duration(elapsed)} elapsed={duration(time.perf_counter() - run_started)} "
            f"remaining={duration(remaining_seconds)} eta_utc={iso_utc(eta)}",
            flush=True,
        )

    print(f"sweep_complete completed={len(rows)}/{len(groups)} finished_utc={iso_utc(utc_now())} results={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
