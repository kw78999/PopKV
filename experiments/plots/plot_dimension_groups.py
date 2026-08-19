from __future__ import annotations

"""Combine four PopKV G_dim kernel runs into one table and comparison plot."""

import argparse
import csv
import json
from pathlib import Path


KERNEL_ORDER = ("g16", "g32", "g64-u32", "g64-u64")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dim_kernel_analysis"))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolved_kernel(metadata: dict) -> str:
    group = int(metadata["dim_group_size"])
    kernel = metadata.get("dim_kernel", "auto")
    return {16: "g16", 32: "g32", 64: "g64-u32"}[group] if kernel == "auto" else kernel


def load_run(path: Path) -> dict:
    metadata_path, summary_path = path / "metadata.json", path / "summary.csv"
    if not metadata_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"{path} must contain metadata.json and summary.csv")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "popkv"]
    if len(rows) != 1:
        raise ValueError(f"{summary_path} must contain exactly one PopKV summary row")
    row = rows[0]
    kernel = resolved_kernel(metadata)
    return {
        "dim_kernel": kernel,
        "dim_group_size": int(metadata["dim_group_size"]),
        "tokens": int(row["tokens"]),
        "perplexity": float(row["perplexity"]),
        "score_ms_per_token": float(row["score_ms_per_token"]),
        "topk_ms_per_token": float(row["topk_ms_per_token"]),
        "gather_ms_per_token": float(row["gather_ms_per_token"]),
        "attention_ms_per_token": float(row["attention_ms_per_token"]),
        "decode_ms_per_token": float(row["decode_ms_per_token"]),
        "k_metadata_encoding_ms_per_token": float(row["retrieval_k_metadata_encoding_ms_per_token"]),
        "update_ms_per_token": float(row["update_ms_per_token"]),
        "generation_step_ms_per_token": float(row["generation_step_ms_per_token"]),
        "tokens_per_second": float(row["generation_tokens_per_second"]),
    }


def make_plot(rows: list[dict], output: Path, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: python -m pip install matplotlib") from exc
    labels = [row["dim_kernel"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = (
        ("score_ms_per_token", "Score latency", "ms/token"),
        ("generation_step_ms_per_token", "Generation latency", "ms/token"),
        ("perplexity", "Perplexity", "PPL (lower is better)"),
    )
    for ax, (key, title, ylabel) in zip(axes, panels):
        values = [row[key] for row in rows]
        bars = ax.bar(labels, values, color=("#4c78a8", "#f58518", "#54a24b", "#e45756"))
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        ax.bar_label(bars, fmt="%.4g", padding=2, fontsize=8)
    figure.savefig(output / "dim_kernel_ablation.png", dpi=dpi)
    figure.savefig(output / "dim_kernel_ablation.pdf")
    plt.close(figure)


def main() -> None:
    args = arguments()
    rows_by_kernel = {}
    for path in args.inputs:
        row = load_run(path)
        kernel = row["dim_kernel"]
        if kernel in rows_by_kernel:
            raise ValueError(f"duplicate dim_kernel={kernel}")
        rows_by_kernel[kernel] = row
    missing = set(KERNEL_ORDER) - set(rows_by_kernel)
    if missing:
        raise ValueError(f"missing kernels={sorted(missing)}")
    rows = [rows_by_kernel[kernel] for kernel in KERNEL_ORDER]
    if len({row["tokens"] for row in rows}) != 1:
        raise ValueError("all runs must contain the same number of evaluated tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dim_kernel_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    make_plot(rows, args.output_dir, args.dpi)
    print("kernel  G_dim  score_ms  generation_ms  PPL")
    for row in rows:
        print(f"{row['dim_kernel']:8s} {row['dim_group_size']:5d} "
              f"{row['score_ms_per_token']:9.4f} {row['generation_step_ms_per_token']:14.4f} "
              f"{row['perplexity']:.4f}")
    print(f"results={args.output_dir}")


if __name__ == "__main__":
    main()
