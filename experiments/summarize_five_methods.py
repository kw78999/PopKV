"""Regenerate PG19 tables with an exhaustive, non-overlapping timing breakdown."""
from __future__ import annotations

import argparse
import csv
import math
from decimal import Decimal
from pathlib import Path

METHODS = (("full", "Full"), ("quest", "QUEST"), ("fier-author", "FIER-author"),
           ("popkv-normal", "PopKV-Normal"), ("popkv-q", "PopKV-Q"), ("popkv-s", "PopKV-S"))
DECODE_STAGES = (
    ("frontend", "Frontend"), ("qkv_rope", "QKV + RoPE"),
    ("score", "Score (query preparation + scoring)"), ("topk", "Top-k retrieval"),
    ("full_gather", "Full K/V concatenation"), ("full_attention", "Full attention"),
    ("sparse_attention_partial", "Sparse attention partial"),
    ("sparse_attention_merge", "Sparse attention merge"),
    ("output_projection", "Output projection"), ("mlp", "MLP"),
    ("final_norm_lm_head", "Final norm + LM head"),
    ("gpu_unattributed", "GPU timing remainder"), ("host_decode_residual", "Host decode remainder"),
)
UPDATE_STAGES = (
    ("kv_update", "Static KV update"), ("retrieval_active_key_staging", "Active-key staging"),
    ("retrieval_k_metadata_encoding", "K metadata encoding"),
    ("retrieval_metadata_write", "Metadata write"), ("retrieval_group_sealing", "Group sealing"),
    ("retrieval_gpu_unattributed", "Retrieval GPU remainder"),
    ("update_host_residual", "Update host remainder"),
)
GENERATION_STAGES = (
    ("score", "Score (query preparation + scoring)"), ("topk", "Top-k retrieval"),
    ("attention", "Attention (dense + sparse)"), ("other_model", "Other model operations"),
    ("decode_overhead", "Decode timing remainder (GPU + host)"),
    ("token_selection", "Next-token selection"), ("update", "K/V + retrieval metadata update"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_time(summary: dict[str, str]) -> float:
    """Legacy runs time preparation + scoring together; newer runs split them."""
    fields = ("retrieval_query_encode_ms_per_token", "retrieval_score_ms_per_token")
    if any(k in summary for k in fields):
        return sum(float(summary[k]) for k in fields)
    return float(summary["score_ms_per_token"])


def generation_breakdown(summary: dict[str, str]) -> dict[str, float]:
    def ms(key: str) -> float:
        return float(summary[f"{key}_ms_per_token"])
    result = {
        "score": score_time(summary), "topk": ms("topk"),
        "attention": sum(ms(k) for k in ("full_attention", "sparse_attention_partial", "sparse_attention_merge")),
        "other_model": sum(ms(k) for k in ("frontend", "qkv_rope", "full_gather", "output_projection", "mlp", "final_norm_lm_head")),
        "decode_overhead": ms("gpu_unattributed") + ms("host_decode_residual"),
        "token_selection": ms("token_selection"), "update": ms("update"),
    }
    measured = ms("generation_step")
    if not math.isclose(sum(result.values()), measured, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Timing does not close for {summary['method']}: {sum(result.values())} != {measured}")
    return result


def additive_table(rows: list[dict], stages: tuple, total_key: str, total_label: str) -> str:
    """Keep printed sums exact without changing any measured component."""
    lines = ["| Stage (ms/token) | " + " | ".join(r["method"] for r in rows) + " |",
             "|---|" + "---:|" * len(rows)]
    for key, label in stages:
        lines.append("| " + label + " | " + " | ".join(f"{r[key]:.3f}" for r in rows) + " |")
    rounding = [Decimal(f"{r[total_key]:.3f}") - sum((Decimal(f"{r[k]:.3f}") for k, _ in stages), Decimal(0)) for r in rows]
    lines.append("| Display rounding adjustment | " + " | ".join(f"{v:.3f}" for v in rounding) + " |")
    lines.append(f"| **{total_label}** | " + " | ".join(f"**{r[total_key]:.3f}**" for r in rows) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.input_dir), Path(args.output_dir)
    summaries = {r["method"]: r for r in read_csv(source / "summary.csv")}
    samples = read_csv(source / "samples.csv")
    unknown = set(summaries) - {m for m, _ in METHODS}
    if unknown or not summaries:
        raise ValueError(f"Invalid method set: {set(summaries)}")
    overall, decode, update, closure, generation, score_rows = [], [], [], [], [], []
    for method, label in METHODS:
        if method not in summaries:
            continue
        s = summaries[method]
        prepack = {int(r["block"]): float(r["prepack_ms"]) for r in samples if r["method"] == method}
        steady = [v for k, v in prepack.items() if k != 0]
        def ms(key: str) -> float:
            return float(s[f"{key}_ms_per_token"])
        overall.append(dict(method=label, perplexity=float(s["perplexity"]),
                            prepack_all_ms_per_block=float(s["prepack_ms"]), prepack_block0_ms=prepack[0],
                            prepack_steady_ms_per_block=sum(steady)/len(steady) if steady else prepack[0],
                            decode_ms_per_token=ms("decode"), token_selection_ms_per_token=ms("token_selection"),
                            update_ms_per_token=ms("update"), generation_ms_per_token=ms("generation_step"),
                            tokens_per_second=float(s["generation_tokens_per_second"])))
        d = {k: score_time(s) if k == "score" else ms(k) for k, _ in DECODE_STAGES}
        u = {k: ms(k) for k, _ in UPDATE_STAGES}
        decode.append(dict(method=label, **d, total=ms("decode")))
        update.append(dict(method=label, **u, total=ms("update")))
        g = generation_breakdown(s)
        generation.append(dict(method=label, **g, total=ms("generation_step")))
        score_rows.append(dict(method=label, score_ms_per_token=g["score"],
                               topk_ms_per_token=g["topk"], search_ms_per_token=g["score"]+g["topk"]))
        errors = dict(decode_error_ms=ms("decode")-sum(d.values()),
                      update_error_ms=ms("update")-sum(u.values()),
                      generation_error_ms=ms("generation_step")-sum(g.values()))
        if any(abs(v) > 1e-6 for v in errors.values()):
            raise ValueError(f"Unaccounted time for {method}: {errors}")
        closure.append(dict(method=label, decode_measured_ms=ms("decode"), decode_reconstructed_ms=sum(d.values()),
                            update_measured_ms=ms("update"), update_reconstructed_ms=sum(u.values()),
                            generation_measured_ms=ms("generation_step"), generation_reconstructed_ms=sum(g.values()), **errors))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("overall",overall),("generation_breakdown",generation),("score_comparison",score_rows),("latency_closure",closure)):
        write_csv(output/f"{name}.csv", rows)
    for name, rows, stages in (("decode_exclusive",decode,DECODE_STAGES),("update_exclusive",update,UPDATE_STAGES)):
        write_csv(output/f"{name}.csv", [dict(method=r["method"],stage=label,ms_per_token=r[k]) for r in rows for k,label in stages])
    quality = ["| Method | PPL | Generation ms/token | token/s |",
               "|---|---:|---:|---:|"]
    for r in overall:
        quality.append(f"| {r['method']} | {r['perplexity']:.4f} | {r['generation_ms_per_token']:.3f} | {r['tokens_per_second']:.3f} |")
    report = ["# PG19 32K: quality and token-generation time", "",
              "Llama-3.1-8B; context 32,768; budget 4,096; 10 blocks × 256 evaluated tokens; BF16; seed 42. Layers 0 and 1 remain dense.", "",
              "## Quality and overall latency", "", *quality, "",
              "## Additive token-generation breakdown", "",
              "Each component is exclusive. Total = score + Top-k + attention + other model operations + decode timing remainder + next-token selection + cache update.", "",
              additive_table(generation, GENERATION_STAGES, "total", "Total token generation"), "",
              "Score includes query preparation and approximate scoring, excluding Top-k. The historical raw data records this as one score interval; newer evaluations record query encoding and scoring separately, which this script adds. A separate historical query-encoding time cannot be recovered.", "",
              "Attention includes dense layers and sparse partial/merge kernels. Other model operations include frontend, QKV/RoPE, full K/V concatenation, output projection, MLP, and final norm/head. The timing remainder retains measured GPU intervals outside named stages and host residual time. Next-token selection chooses the output token; Top-k chooses K/V positions.", "",
              "The display rounding adjustment only reconciles independently rounded numbers (0.001 ms precision); it is not a measured stage. CSV values retain full precision. Closure is checked within 1e-6 ms/token before writing tables.", "",
              "Prefix prefill and one-time metadata prepack are outside this decode-cycle total. Prepack measurements are preserved in overall.csv (ms/block); token/s = 1000 / total token-generation ms.", "",
              "## Decode stages (including timing remainders)", "", additive_table(decode, DECODE_STAGES,"total","Decode total"), "",
              "## Cache-update stages", "", additive_table(update,UPDATE_STAGES,"total","Update total"), "",
              "The decode and update tables expand components of the generation table; do not add their totals a second time. Numerical checks are in latency_closure.csv.", ""]
    (output/'report.md').write_text('\n'.join(report))
    print(output)


if __name__ == "__main__":
    main()
