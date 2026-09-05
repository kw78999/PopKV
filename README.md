# PopKV

Official implementation of **PopKV**, a training-free KV-cache retrieval method for long-context autoregressive decoding.

[Paper](https://openreview.net/forum?id=7kA636r4L1) · [Minseok Choi](https://openreview.net/profile?id=~Minseok_Choi6) · [Geonhee Jang](https://openreview.net/profile?id=~Geonhee_Jang1)

PopKV encodes query/key retrieval metadata as sign and magnitude bit planes, ranks cached tokens with CUDA popcount kernels, and computes attention over the selected full-precision K/V entries. It reduces retrieval and attention work while retaining the original K/V cache.

## Methods

| Method | Retrieval | Token group | Dimension group |
|---|---|---:|---:|
| Full | Dense attention | — | — |
| QUEST | Page min/max bounds; page size 16 | — | — |
| FIER | Sequence-grouped one-bit retrieval | 32 | — |
| PopKV-Q (Ours) | Popcount retrieval; quality preset | 128 | 16 |
| PopKV-S (Ours) | Popcount retrieval; speed preset | 64 | 64 |

All published comparisons use these five methods. `popkv-normal` (token group 32, dimension group 32) remains available for ablations. Sparse methods keep layers 0 and 1 dense. Each query head selects its own tokens, including for GQA; the current token reserves one budget slot. Sparse attention uses 256-token partial reductions followed by a merge.

For channel `i` in dimension group `g`, the approximation is:

```text
x_hat[i] = sign[i] * (low[g] + magnitude[i] * delta[g])
```

`low` and `delta = high - low` are representative magnitudes. The approximate dot product combines four signed agreement counts computed with 32-bit popcount, without reconstructing dense Q/K vectors. Final attention uses original-precision K/V values.

## Installation

Use Linux, an NVIDIA GPU, a CUDA toolkit with `nvcc`, and a CUDA-enabled PyTorch installation. Install PyTorch for your CUDA environment before the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -r requirements.txt
```

The CUDA extension compiles on first use through PyTorch's extension loader. Authenticate for gated checkpoints with `hf auth login` or set `HF_TOKEN` (see [.env.example](.env.example)). Dataset caches, logs, local experiment outputs, and credentials are excluded from Git.

The recorded LongBench environment used an RTX A6000 (48 GB), Python 3.11.10, PyTorch 2.4.1+cu124, and Transformers 4.57.6. See [environment](results/longbench_950_b4096/environment.txt) and [package versions](results/longbench_950_b4096/pip_freeze.txt). The historical PG19 metadata does not contain an equivalent hardware/software snapshot; do not treat the LongBench snapshot as a verified PG19 environment.

## PG19: 32K quality and token-generation latency

**Model:** `meta-llama/Llama-3.1-8B` · **Context:** 32,768 · **Budget:** 4,096 · **Evaluation:** 10 blocks × 256 tokens per method · **Precision:** BF16 · **Seed:** 42.

PG19 PPL is sampled next-token PPL from fixed blocks. Latency is the measured decode cycle after prefix prefill; token-generation time includes decode, next-token selection, and cache updates.

| Method | PPL ↓ | Generation ms/token ↓ | token/s ↑ |
|---|---:|---:|---:|
| Full | 11.1416 | 170.742 | 5.857 |
| QUEST | 13.8755 | 42.157 | 23.721 |
| FIER | 11.1880 | 45.581 | 21.939 |
| PopKV-Q (Ours) | 11.2354 | 43.888 | 22.785 |
| PopKV-S (Ours) | 11.2626 | 43.011 | 23.250 |

### Where token-generation time goes

Each column below is one method. **Add the component rows to obtain the total at the bottom.** All times are ms/token, averaged over the same 2,560 evaluated tokens per method.

| Stage (ms/token) | Full | QUEST | FIER | PopKV-Q (Ours) | PopKV-S (Ours) |
|---|---:|---:|---:|---:|---:|
| Score (query preparation + scoring) | 0.000 | 1.006 | 3.227 | 1.365 | 0.813 |
| Top-k retrieval | 0.000 | 0.882 | 1.844 | 1.814 | 1.769 |
| Attention (dense + sparse) | 124.120 | 11.221 | 11.349 | 11.285 | 11.288 |
| Other model operations | 45.424 | 27.149 | 27.230 | 27.165 | 27.074 |
| Decode timing remainder (GPU + host) | 0.438 | 0.488 | 0.497 | 0.482 | 0.487 |
| Next-token selection | 0.069 | 0.060 | 0.060 | 0.060 | 0.060 |
| K/V + retrieval metadata update | 0.691 | 1.350 | 1.373 | 1.717 | 1.519 |
| Display rounding adjustment | 0.000 | 0.001 | 0.001 | 0.000 | 0.001 |
| **Total token generation** | **170.742** | **42.157** | **45.581** | **43.888** | **43.011** |

- **Score** includes query preparation and approximate scoring, and excludes Top-k. The historical run records these together; the current evaluator records query encoding and scoring separately, and the summarizer adds them. The historical file cannot provide a separate query-encoding measurement.
- **Top-k retrieval** selects cached K/V positions. **Next-token selection** chooses the output token from model logits; these are separate operations.
- **Attention** includes dense attention in the first two layers and sparse partial/merge attention elsewhere.
- **Other model operations** include frontend, QKV/RoPE, full K/V concatenation, output projection, MLP, and final norm/LM head.
- **Decode timing remainder** includes GPU time outside named intervals and host residual time. It is retained so the accounting matches the measured total.
- **Display rounding adjustment** reconciles the independently rounded entries at 0.001 ms precision; it is not a measured operation. Full-precision CSV totals close within 0.000001 ms/token.

Prefix prefill and one-time metadata prepack are outside this decode-cycle total. Prepack measurements remain in [overall.csv](results/five_methods_split_attention_32k_b4096_t256_blocks10/overall.csv), in ms/block. Throughput is `1000 / total token-generation ms`.

### Score-time comparison

FIER takes **3.227 ms/token** for score, compared with **1.365** for PopKV-Q and **0.813** for PopKV-S: **2.36×** and **3.97×** faster score stages (57.7% and 74.8% less time). QUEST takes **1.006 ms/token**, placing PopKV-Q above QUEST and PopKV-S below it. These are score-stage comparisons; end-to-end gains also depend on Top-k, attention, model operations, and cache updates.

PopKV-Q/S achieve approximately **3.89× / 3.97×** Full's token-generation throughput in this run. QUEST has the shortest total time; FIER has the lowest sparse-method PPL.

[Detailed additive decode/update tables](results/five_methods_split_attention_32k_b4096_t256_blocks10/report.md) · [Full-precision timing components](results/five_methods_split_attention_32k_b4096_t256_blocks10/generation_breakdown.csv) · [Closure checks](results/five_methods_split_attention_32k_b4096_t256_blocks10/latency_closure.csv)

### Reproduce or rebuild PG19 tables

```bash
python -m experiments.evaluate_pg19 \
  --model meta-llama/Llama-3.1-8B \
  --methods full,quest,fier,popkv-q,popkv-s \
  --context-length 32768 --budget 4096 \
  --generate-tokens 256 --num-blocks 10 --seed 42 \
  --dtype bfloat16 --attention sdpa --full-layers 0,1 \
  --output-dir outputs/pg19_32k

python -m experiments.summarize_five_methods \
  --input-dir outputs/pg19_32k --output-dir outputs/pg19_32k_tables
```

Regenerate the published tables from the included measurements without loading a model:

```bash
python -m experiments.summarize_five_methods \
  --input-dir results/five_methods_split_attention_32k_b4096_t256_blocks10/raw \
  --output-dir outputs/rebuilt_pg19_tables
```

The loader caches PG19 test books under `data/pg19/`. Use `--text-file PATH` for a local corpus. Re-running the current kernels produces a new measurement; the included raw files remain the evidence for the historical numbers.

## PG19: context-length sweep

The [4–128K report](results/context_sweep_4k_128k_final/report.md) uses **4 blocks × 256 evaluated tokens** at budget 4,096, with 1K steps (`K = 1,024 tokens`). This is a separate experiment from the 10-block table above.

<img width="409" height="240" alt="image" src="https://github.com/user-attachments/assets/b8d3138f-af27-4610-8bb7-7ae84412693a" />

Score computation time increases with context length across all methods. PopKV achieves substantially lower scoring overhead than FIER while remaining competitive with QUEST, demonstrating better scalability for long-context retrieval.

[Wide table](results/context_sweep_4k_128k_final/final_context_table.csv) · [All metrics and source labels](results/context_sweep_4k_128k_final/combined_long.csv) · [Reproduction instructions](results/README.md)

## LongBench: five-task quality evaluation

**Model:** `NousResearch/Meta-Llama-3.1-8B-Instruct` · **Context truncation setting:** 32,768 · **Budget:** 4,096 · **Precision:** BF16 · **Seed:** 42.

The release evaluates all valid samples of five selected tasks: Qasper (200), MultiFieldQA-en (150), HotpotQA (200), GovReport (200), and TriviaQA (200). Each method has **950 successful samples**, for **4,750 retained predictions**. This is a five-task subset of LongBench.

| Method | Qasper | MultiFieldQA-en | HotpotQA | GovReport | TriviaQA | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full | 45.982 | 53.050 | 55.354 | 19.917 | 91.645 | 53.190 |
| QUEST | 47.277 | 53.550 | 55.369 | 20.111 | 89.669 | 53.195 |
| FIER | 46.146 | 53.630 | 56.058 | 19.972 | 91.478 | 53.457 |
| PopKV-Q | 46.042 | 53.553 | 55.258 | 19.931 | 91.478 | 53.252 |
| PopKV-S | 45.602 | 53.929 | 54.891 | 20.042 | 91.469 | 53.187 |

Scores are percentages: token F1 for the four QA tasks and ROUGE-L for GovReport. Average is the unweighted mean of the five task scores. Full-relative retention in the detailed report is the mean of per-task score ratios, not the ratio of overall averages. Small score differences should not be interpreted as demonstrated superiority without uncertainty estimates.

The retained predictions are unchanged historical measurements, reaggregated for this five-task release. Prompts are truncated before chat-template application; template tokens can make the final input slightly longer than the truncation setting.

[Detailed scores and runtime](results/longbench_950_b4096/README.md) · [Experiment configuration](results/longbench_950_b4096/experiment_config.json) · [Sample manifest](results/longbench_950_b4096/sample_manifest.json) · [Raw predictions and measurements](results/longbench_950_b4096/raw_results.jsonl)

```bash
python -m experiments.evaluate_longbench \
  --model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --methods full,quest,fier,popkv-q,popkv-s \
  --datasets qasper,multifieldqa_en,hotpotqa,gov_report,triviaqa \
  --budgets 4096 --all-samples --max-context 32768 \
  --dtype bfloat16 --device-map cuda:0 --attention sdpa \
  --full-layers 0,1 --seed 42 --validate \
  --hard-limit-minutes 1800 --stop-reserve-minutes 15 \
  --output-dir outputs/longbench_950
```

Re-running the same command resumes completed samples. To rebuild only the included score/runtime tables, without loading a model:

```bash
python -m experiments.evaluate_longbench \
  --aggregate-only --output-dir results/longbench_950_b4096
```

## Repository layout

```text
popkv.py                    Caches, retrieval, decoder, and reference implementations
popkv_cuda.py               CUDA extension loader and Python interface
csrc/                       CUDA/C++ packing, update, and scoring kernels
sparse_kernels.py           QUEST and indexed sparse attention kernels
fier_triton.py              FIER packing and scoring
experiments/                Evaluation, aggregation, sweeps, and cost analysis
experiments/benchmarks/     Encoding and dimension-group microbenchmarks
experiments/plots/          Ablation plotting utilities
tests/                     Correctness and published-result consistency checks
results/                   Final tables, reproducibility inputs, and provenance
```

See [results/README.md](results/README.md) for the artifact inventory and table-rebuild commands. Resource-cost results are analytical operation counts and logical tensor traffic, not measured hardware-counter DRAM traffic. K/V remains BF16 and retrieval metadata adds storage.

## Validation and ablations

```bash
python -m unittest discover -s tests -v
python -m experiments.evaluate_pg19 --dry-run
python -m experiments.evaluate_longbench --dry-run
python -m experiments.sweep_token_groups --dry-run
python -m experiments.benchmarks.benchmark_encoding --dry-run
python -m experiments.benchmarks.benchmark_dimension_groups --dry-run
```

GPU tests require CUDA; they are skipped when CUDA is unavailable. Token-group and dimension-group analysis utilities remain available under `experiments/`; intermediate local runs are excluded from the published result set.

## Scope

This implementation targets Llama-style autoregressive decoding and supports per-query-head retrieval with MHA/GQA/MQA mappings. Prefix prefill remains dense. The method compresses retrieval metadata rather than the final attention values. Absolute timings depend on hardware, software, compilation, and profiling overhead; comparisons should use the same measurement setup.

For citation metadata, see the [paper page](https://openreview.net/forum?id=7kA636r4L1).
