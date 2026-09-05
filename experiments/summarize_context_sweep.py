from __future__ import annotations

"""Merge low-context and existing high-context PG19 sweeps."""

import argparse
import csv
from pathlib import Path


METHODS = ("full", "quest", "fier", "popkv-q", "popkv-s")
METHOD_LABELS = {
    "full": "Full",
    "quest": "Quest (GQA fused)",
    "fier": "FIER",
    "popkv-q": "PopKV-Q",
    "popkv-s": "PopKV-S",
}
METHOD_SLUGS = {method: method.replace("-", "_") for method in METHODS}
CHECKPOINTS_KIB = (4, 8, 16, 32, 64, 96, 125, 128)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--low-root",
        default="outputs/context_sweep_c4k_31k_step1k_b4096_blocks4_t256",
    )
    parser.add_argument(
        "--quest-root",
        default="outputs/context_sweep_quest_gqa_fused_c32k_128k_step1k_b4096_blocks4_t256",
    )
    parser.add_argument(
        "--previous-root",
        default="outputs/context_sweep_c32k_128k_step1k_b4096_blocks4_t256",
    )
    parser.add_argument("--output-dir", default="results/context_sweep_4k_128k_final")
    parser.add_argument("--context-start-k", type=int, default=4)
    parser.add_argument("--context-end-k", type=int, default=128)
    parser.add_argument("--expected-tokens", type=int, default=1024)
    return parser.parse_args()


def read_complete_rows(case_dir: Path, expected_tokens: int) -> dict[str, dict[str, str]]:
    summary = case_dir / "summary.csv"
    if not summary.exists():
        return {}
    with summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["method"]: row
        for row in rows
        if int(row.get("tokens", 0)) == expected_tokens
    }


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(value: str | None, digits: int = 2) -> str:
    return "—" if value in (None, "") else f"{float(value):.{digits}f}"


def main() -> None:
    args = arguments()
    if args.context_end_k < args.context_start_k:
        raise ValueError("context end must be greater than or equal to context start")

    low_root = Path(args.low_root)
    quest_root = Path(args.quest_root)
    previous_root = Path(args.previous_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, str]] = []
    wide: list[dict[str, str]] = []
    available: dict[tuple[int, str], dict[str, str]] = {}
    summary_fields: list[str] | None = None

    for context_kib in range(args.context_start_k, args.context_end_k + 1):
        context = context_kib * 1024
        low = read_complete_rows(low_root / f"c{context}", args.expected_tokens)
        latest = read_complete_rows(quest_root / f"c{context}", args.expected_tokens)
        previous = read_complete_rows(previous_root / f"c{context}", args.expected_tokens)
        selected: dict[str, tuple[dict[str, str], str]] = {}
        if context_kib < 32:
            for method in METHODS:
                if method in low:
                    selected[method] = (low[method], "new_low_context_sweep")
        else:
            if "quest" in latest:
                selected["quest"] = (latest["quest"], "latest_quest_gqa_fused")
            for method in ("full", "fier", "popkv-q", "popkv-s"):
                if method in previous:
                    selected[method] = (previous[method], "previous_multi_method")

        wide_row = {
            "context_kib": str(context_kib),
            "context_length": str(context),
        }
        for method in METHODS:
            slug = METHOD_SLUGS[method]
            if method not in selected:
                for metric in ("perplexity", "decode_ms_per_token", "generation_tokens_per_second"):
                    wide_row[f"{slug}_{metric}"] = ""
                continue
            row, source = selected[method]
            summary_fields = summary_fields or list(row)
            available[(context_kib, method)] = row
            combined.append({
                "context_kib": str(context_kib),
                "context_length": str(context),
                "source": source,
                **row,
            })
            for metric in ("perplexity", "decode_ms_per_token", "generation_tokens_per_second"):
                wide_row[f"{slug}_{metric}"] = row[metric]
        wide.append(wide_row)

    if summary_fields is None:
        raise RuntimeError("no complete input rows found")

    long_fields = ["context_kib", "context_length", "source", *summary_fields]
    write_csv(output / "combined_long.csv", combined, long_fields)
    write_csv(output / "final_context_table.csv", wide, list(wide[0]))

    checkpoint_rows: list[dict[str, str]] = []
    for context_kib in CHECKPOINTS_KIB:
        for method in METHODS:
            row = available.get((context_kib, method))
            checkpoint_rows.append({
                "context_kib": str(context_kib),
                "method": METHOD_LABELS[method],
                "perplexity": "" if row is None else row["perplexity"],
                "decode_ms_per_token": "" if row is None else row["decode_ms_per_token"],
                "generation_tokens_per_second": "" if row is None else row["generation_tokens_per_second"],
            })
    write_csv(
        output / "checkpoint_summary.csv",
        checkpoint_rows,
        ["context_kib", "method", "perplexity", "decode_ms_per_token", "generation_tokens_per_second"],
    )

    context_count = args.context_end_k - args.context_start_k + 1
    coverage = {
        method: sum(
            (context_kib, method) in available
            for context_kib in range(args.context_start_k, args.context_end_k + 1)
        )
        for method in METHODS
    }
    coverage_text = ", ".join(
        f"{METHOD_LABELS[method]} {coverage[method]}/{context_count}"
        for method in METHODS
    )
    report = [
        f"# PG19 {args.context_start_k}–{args.context_end_k}K context sweep: consolidated results",
        "",
        "Configuration: Llama-3.1-8B, budget 4096, 4 PG19 blocks, 256 evaluated tokens per block, seed 42.",
        "",
        "Source policy: 4–31K uses the new low-context multi-method sweep. From 32K onward, Quest uses the latest GQA-fused sweep while Full, FIER, PopKV-Q, and PopKV-S use the earlier multi-method sweep. Results with fewer than 1024 evaluated tokens are excluded. One-block experiments are not mixed into this table.",
        "",
        f"Coverage: {coverage_text}.",
        "",
        "| Context | Method | PPL ↓ | Decode ms/token ↓ | Tokens/s ↑ |",
        "|---:|:---|---:|---:|---:|",
    ]
    for item in checkpoint_rows:
        report.append(
            f"| {item['context_kib']}K | {item['method']} | "
            f"{display(item['perplexity'])} | {display(item['decode_ms_per_token'])} | "
            f"{display(item['generation_tokens_per_second'])} |"
        )
    report.extend([
        "",
        "`—` means that no complete, configuration-compatible measurement is available.",
        "",
        "Files: `final_context_table.csv` is the one-row-per-context final table; `combined_long.csv` preserves all available metrics and source provenance; `checkpoint_summary.csv` contains the compact checkpoints shown above.",
        "",
    ])
    (output / "report.md").write_text("\n".join(report))

    print(f"coverage={coverage_text}")
    print(f"combined_rows={len(combined)}")
    print(f"results={output}")


if __name__ == "__main__":
    main()
