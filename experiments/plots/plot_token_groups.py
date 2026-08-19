from __future__ import annotations

"""Aggregate PG19 blocks by token index and plot PopKV metrics."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-blocks", type=int, default=8)
    parser.add_argument("--expected-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/token_group_plots"))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def load_run(path: Path, expected_blocks: int, expected_tokens: int):
    metadata_path, samples_path = path / "metadata.json", path / "samples.csv"
    if not metadata_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(f"{path} must contain metadata.json and samples.csv")
    group_size = int(json.loads(metadata_path.read_text(encoding="utf-8"))["group_size"])
    by_step: dict[int, dict[int, tuple[float, float]]] = {}
    with samples_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "popkv":
                continue
            block, step = int(row["block"]), int(row["step"])
            value = (float(row["retrieval_k_metadata_encoding_ms"]), float(row["nll"]))
            if block in by_step.setdefault(step, {}):
                raise ValueError(f"duplicate block={block}, step={step} in {samples_path}")
            by_step[step][block] = value

    expected_steps = set(range(expected_tokens))
    if set(by_step) != expected_steps:
        missing = sorted(expected_steps - set(by_step))
        extra = sorted(set(by_step) - expected_steps)
        raise ValueError(f"{path}: token-index mismatch; missing={missing[:10]}, extra={extra[:10]}")

    output = []
    for step in range(expected_tokens):
        values = by_step[step]
        if len(values) != expected_blocks or set(values) != set(range(expected_blocks)):
            raise ValueError(
                f"{path}: step {step} has blocks={sorted(values)}, expected 0..{expected_blocks - 1}"
            )
        latency = [values[block][0] for block in range(expected_blocks)]
        nll = [values[block][1] for block in range(expected_blocks)]
        mean_nll, nll_se = statistics.mean(nll), standard_error(nll)
        output.append({
            "group_size": group_size,
            "token_index": step + 1,
            "blocks": expected_blocks,
            "k_metadata_encoding_mean_ms": statistics.mean(latency),
            "k_metadata_encoding_se_ms": standard_error(latency),
            "mean_nll": mean_nll,
            "nll_se": nll_se,
            "perplexity": math.exp(mean_nll),
            "perplexity_lower_1se": math.exp(mean_nll - nll_se),
            "perplexity_upper_1se": math.exp(mean_nll + nll_se),
        })
    return group_size, output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows_by_group: dict[int, list[dict]], output: Path, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: python -m pip install matplotlib") from exc
    colors = plt.get_cmap("tab10")

    def latency_panel(ax) -> None:
        for index, (group, rows) in enumerate(sorted(rows_by_group.items())):
            x = [row["token_index"] for row in rows]
            y = [row["k_metadata_encoding_mean_ms"] for row in rows]
            se = [row["k_metadata_encoding_se_ms"] for row in rows]
            color = colors(index)
            ax.plot(x, y, label=f"g{group}", color=color, linewidth=1.35)
            ax.fill_between(x, [a - b for a, b in zip(y, se)],
                            [a + b for a, b in zip(y, se)],
                            color=color, alpha=0.10, linewidth=0)
        ax.set(title="K metadata encoding latency", xlabel="Generated token index",
               ylabel="Latency (ms/token)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)

    def perplexity_panel(ax) -> None:
        for index, (group, rows) in enumerate(sorted(rows_by_group.items())):
            x = [row["token_index"] for row in rows]
            y = [row["perplexity"] for row in rows]
            lower = [row["perplexity_lower_1se"] for row in rows]
            upper = [row["perplexity_upper_1se"] for row in rows]
            color = colors(index)
            ax.plot(x, y, label=f"g{group}", color=color, linewidth=1.35)
            ax.fill_between(x, lower, upper, color=color, alpha=0.08, linewidth=0)
        ax.set(title="Per-token-index perplexity", xlabel="Generated token index",
               ylabel="PPL = exp(mean NLL across blocks)")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.legend(ncol=3)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    latency_panel(axes[0])
    perplexity_panel(axes[1])
    figure.savefig(output / "token_group_trajectories.png", dpi=dpi)
    figure.savefig(output / "token_group_trajectories.pdf")
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    latency_panel(ax)
    figure.savefig(output / "k_metadata_encoding_latency.png", dpi=dpi)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    perplexity_panel(ax)
    figure.savefig(output / "perplexity_by_token_index.png", dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = arguments()
    if min(args.expected_blocks, args.expected_tokens, args.dpi) <= 0:
        raise SystemExit("expected blocks, expected tokens, and dpi must be positive")
    rows_by_group = {}
    for path in args.inputs:
        group, rows = load_run(path, args.expected_blocks, args.expected_tokens)
        if group in rows_by_group:
            raise ValueError(f"duplicate group_size={group}")
        rows_by_group[group] = rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flattened = [row for group in sorted(rows_by_group) for row in rows_by_group[group]]
    write_csv(args.output_dir / "per_token_index_metrics.csv", flattened)
    make_plots(rows_by_group, args.output_dir, args.dpi)
    print(f"groups={','.join(str(group) for group in sorted(rows_by_group))}")
    print(f"results={args.output_dir}")


if __name__ == "__main__":
    main()
