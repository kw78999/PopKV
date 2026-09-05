# Published experiment artifacts

The release contains two PG19 experiments, a five-task LongBench evaluation, and an analytical resource-cost appendix. Raw measurements are retained; summary files can be rebuilt locally.

| Directory | Contents |
|---|---|
| `five_methods_split_attention_32k_b4096_t256_blocks10/` | 32K quality, additive latency, score comparison, and closure tables; original configuration, summaries, and 12,800 token records in `raw/` |
| `context_sweep_4k_128k_final/` | Consolidated 4–128K results and source labels; per-context summaries/configurations in `inputs/` |
| `longbench_950_b4096/` | Five tasks, 950 samples per method, 4,750 raw predictions; scores, runtime, manifest, environment, and implementation provenance |
| `resource_cost_32k_b4096_t256_blocks10_split_attention/` | Analytical operation counts, logical tensor traffic, and metadata storage; not hardware-counter measurements |

`SHA256SUMS` records hashes of the included raw evidence and configuration files. Verify from the repository root with `sha256sum -c results/SHA256SUMS`.

## Rebuild the PG19 32K tables

```bash
python -m experiments.summarize_five_methods \
  --input-dir results/five_methods_split_attention_32k_b4096_t256_blocks10/raw \
  --output-dir outputs/rebuilt_pg19_tables
```

The summarizer accepts both historical combined-score intervals and current separate query-encoding/scoring intervals. It rejects timing discrepancies above 1e-6 ms/token. The report explicitly includes timing residuals and display-rounding adjustments.

## Rebuild the consolidated context tables

```bash
python -m experiments.summarize_context_sweep \
  --low-root results/context_sweep_4k_128k_final/inputs/low \
  --quest-root results/context_sweep_4k_128k_final/inputs/quest \
  --previous-root results/context_sweep_4k_128k_final/inputs/previous \
  --output-dir outputs/rebuilt_context_tables
```

Only measurements with exactly 1,024 evaluated tokens are selected. Original case summaries are retained even when incomplete, so missing coverage can be audited. For 4–31K, the low-context source supplies all methods. For 32–128K, the QUEST source supplies QUEST and the previous source supplies the other methods.

For a new sweep with the current code:

```bash
python -m experiments.run_context_sweep \
  --context-start-k 4 --context-end-k 128 --context-step-k 1 \
  --model meta-llama/Llama-3.1-8B \
  --methods full,quest,fier,popkv-q,popkv-s \
  --budget 4096 --num-blocks 4 --generate-tokens 256 \
  --prefill-chunk-size 8192 --seed 42 \
  --output-root outputs/new_context_sweep
```

A new sweep is a new experiment, not a reconstruction of the historical mixed-source timing run. It may stop at the device's memory limit; use a new output root for any method subset resumed beyond that limit.

## Rebuild LongBench tables

```bash
python -m experiments.evaluate_longbench \
  --aggregate-only --output-dir results/longbench_950_b4096
```

This uses the saved configuration and predictions without loading a model or downloading datasets. Read [provenance](longbench_950_b4096/PROVENANCE.md) for the scope of the historical implementation patch.

## Historical PG19 evidence

The PG19 32K raw metadata identifies the model, precision, attention backend, seed, block count, and retrieval settings. It does not record a model revision, corpus hash, or GPU/software snapshot. The release preserves these records as observed rather than assigning unverified environment details retrospectively. The resource-cost appendix is analytical and is not used to fill gaps in measured latency.

Intermediate PG19 optimization, smoke, pilot, and ablation outputs are kept locally under ignored `outputs/` paths; they are outside this published result set.
