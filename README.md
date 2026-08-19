# PopKV

Official implementation of **PopKV**, a training-free KV-cache retrieval method for efficient long-context autoregressive decoding.

**Paper:** [PopKV](https://openreview.net/forum?id=7kA636r4L1)

**Authors:** [Minseok Choi](https://openreview.net/profile?id=~Minseok_Choi6), [Geonhee Jang](https://openreview.net/profile?id=~Geonhee_Jang1)

PopKV compresses query/key retrieval metadata into sign and magnitude bit planes, retains two representative magnitudes per dimension group, and ranks cached tokens with fused CUDA popcount kernels. Final attention uses the selected full-precision K/V entries.

The repository provides four comparable decoding methods:

- `full`: dense full attention.
- `quest`: page-level min/max upper-bound retrieval with fused Triton scoring and page selection.
- `fier`: FIER-style sequence-grouped one-bit retrieval with fused Triton scoring.
- `popkv`: representative-weighted bit-popcount retrieval with specialized CUDA kernels.


## Highlights

- Training-free and compatible with full-precision K/V caches.
- Per-query-head token retrieval for Llama-style MHA, GQA, and MQA mappings.
- Specialized PopKV kernels for token groups 8, 16, 32, 64, and 128 and dimension groups 16, 32, and 64.
- Fused QUEST scoring and selection kernels.
- Reproducible PG19 latency/PPL and LongBench quality experiments.

## Method

For channel `i` in dimension group `g`, PopKV reconstructs a query or key approximation as

```text
x_hat[i] = sign[i] * (low[g] + magnitude[i] * delta[g])
```

`sign` and `magnitude` are binary indicators. `low` is the mean absolute value of low-magnitude channels and `delta = high - low`. The approximate dot product is evaluated without reconstructing dense vectors:

```text
score = sum_g (q_low*k_low*Csign
             + q_delta*k_low*Cq
             + q_low*k_delta*Ck
             + q_delta*k_delta*Cqk) / sqrt(head_dim)
```

The signed agreement counts `Csign`, `Cq`, `Ck`, and `Cqk` are computed with 32-bit popcount operations. Retrieval is independent for every query head, and the newest token always reserves one budget position.

## Repository layout

```text
popkv.py                         Core caches, retrieval methods, decoder, and references
popkv_cuda.py                    JIT loader and Python interface for PopKV CUDA kernels
sparse_kernels.py                Sparse attention CUDA/Triton kernels
fier_triton.py                   FIER metadata packing and scoring kernels
csrc/popkv_ext.{cpp,cu}          PopKV packing, update, query encoding, and score kernels
experiments/evaluate_pg19.py     PG19 PPL and detailed latency evaluation
experiments/evaluate_longbench.py LongBench quality evaluation
experiments/sweep_token_groups.py Token-group sweep
experiments/benchmarks/          CUDA microbenchmarks
experiments/plots/               Ablation plotting utilities
tests/                           CPU/CUDA correctness tests
results/                         Compact machine-readable paper results
```

## Installation

The kernels require Linux, an NVIDIA GPU, a CUDA toolkit with `nvcc`, and a CUDA-enabled PyTorch installation.

```bash
cd PopKV
python -m venv .venv
source .venv/bin/activate

# Install the CUDA-enabled PyTorch build appropriate for your system first.
pip install torch
pip install -r requirements.txt
```

The CUDA extension is compiled just in time on the first PopKV call. Build artifacts are stored in PyTorch's extension cache, not in this repository.

### Hugging Face authentication

The Llama checkpoints used by the experiments are gated. Request access and authenticate with either:

```bash
hf auth login
```

or:

```bash
export HF_TOKEN=hf_your_read_token
```

Tokens, model caches, datasets, generated outputs, and logs are excluded by `.gitignore`.

## Validation

Run the complete correctness suite:

```bash
python -m unittest discover -s tests -v
```

Validate experiment configurations without loading a model:

```bash
python -m experiments.evaluate_pg19 --dry-run
python -m experiments.evaluate_longbench --dry-run
python -m experiments.sweep_token_groups --dry-run
```

## PG19 32K experiment

The current comparison uses:

- Model: `meta-llama/Llama-3.1-8B`
- Context length: 32,768
- Retrieval budget: 4,096
- Generated/evaluated tokens: 128 per block
- Blocks: 10 non-overlapping PG19 blocks
- Total evaluated tokens: 1,280 per method
- Seed: 42
- Sparse layers: all except layers 0 and 1
- GPU: NVIDIA RTX A6000 48 GB
- PyTorch 2.4.1+cu124, CUDA toolkit 12.4, Triton 3.0.0

The two PopKV operating points are:

- **PopKV-Q**: quality-first, `tok-G=128`, `dim-G=16`, `g16` score kernel.
- **PopKV-S**: speed-first, `tok-G=64`, `dim-G=64`, `g64-u32` score kernel.

### End-to-end results

Prepack is the one-time mean cost per block. All other latency columns are milliseconds per generated token.

| Method | PPL ↓ | Prepack | Decode | Update | Token select | Token cycle | Generation step ↓ | TPS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | **9.532** | **0.290** | 191.838 | **0.669** | 0.074 | 192.507 | 192.581 | 5.193 |
| QUEST | 11.227 | 61.962 | **60.290** | 1.280 | **0.061** | **61.570** | **61.631** | **16.226** |
| FIER | 9.618 | 70.092 | 68.402 | 1.285 | 0.062 | 69.687 | 69.749 | 14.337 |
| PopKV-Q | **9.618** | 52.871 | 63.466 | 1.705 | 0.065 | 65.171 | 65.235 | 15.329 |
| PopKV-S | 9.780 | **40.385** | 62.233 | 1.430 | 0.063 | 63.663 | 63.726 | 15.692 |

`token cycle = decode + update`; `generation step = token cycle + token selection`.

### Decode-stage latency

| Method | Frontend | QKV+RoPE | Score | Top-k | Gather | Attention | Output proj. | MLP | Final norm/head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.408 | 5.073 | 0 | 0 | 20.290 | 140.938 | 2.139 | 20.711 | 1.798 |
| QUEST | **0.365** | 4.890 | **1.099** | **0.856** | 1.292 | 26.837 | 2.115 | 20.519 | 1.796 |
| FIER | 0.368 | 4.915 | 8.704 | 2.461 | 1.300 | 25.653 | 2.118 | 20.559 | 1.796 |
| PopKV-Q | 0.381 | 4.888 | 4.172 | 2.426 | **1.291** | **25.354** | 2.114 | 20.520 | **1.796** |
| PopKV-S | 0.366 | **4.870** | 2.927 | 2.414 | 1.291 | 25.435 | **2.110** | **20.514** | 1.796 |

Gather and attention stage boundaries differ across methods, so compare `gather + attention`, not either stage alone.

| Method | Score+Top-k ↓ | Gather+Attention ↓ | Sparse core total ↓ |
|---|---:|---:|---:|
| Full | 0 | 161.228 | 161.228 |
| QUEST | **1.956** | 28.129 | **30.084** |
| FIER | 11.165 | 26.953 | 38.117 |
| PopKV-Q | 6.598 | **26.645** | 33.243 |
| PopKV-S | 5.341 | 26.726 | 32.067 |

### Update-stage latency

| Method | KV update | Retrieval update | Active staging | Metadata encode | Metadata write | Group sealing | Total update ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.626 | 0 | 0 | 0 | 0 | 0 | **0.669** |
| QUEST | 0.613 | **0.636** | **0.379** | **0.175** | 0.010 | 0.022 | 1.280 |
| FIER | 0.614 | 0.640 | 0.381 | 0.180 | 0.010 | 0.019 | 1.285 |
| PopKV-Q | 0.644 | 1.033 | 0.410 | 0.587 | **0.003** | **0.004** | 1.705 |
| PopKV-S | **0.595** | 0.808 | 0.395 | 0.373 | 0.004 | 0.005 | 1.430 |

Decode GPU component closure is above 99.98% for every method. The complete 43-column result, including GPU/host residual and closure fields, is available at [`results/pg19_32k_b4096_t128_blocks10.csv`](results/pg19_32k_b4096_t128_blocks10.csv).

### Interpretation

- PopKV-Q matches FIER quality (`PPL 9.618`) while reducing generation-step latency by 6.47% and increasing throughput by 6.92%.
- PopKV-S is only 3.40% slower than QUEST but has substantially better PPL (`9.780` versus `11.227`).
- PopKV-S is 3.02x faster than Full attention at the generation-step level.
- PopKV's main remaining optimization targets are score/top-k and metadata-update encoding.

### Reproduction commands

Run the shared baselines:

```bash
python -m experiments.evaluate_pg19 \
  --model meta-llama/Llama-3.1-8B \
  --methods full,quest,fier \
  --context-length 32768 \
  --budget 4096 \
  --generate-tokens 128 \
  --num-blocks 10 \
  --seed 42 \
  --output-dir outputs/pg19_32k/baselines
```

Run PopKV-Q:

```bash
python -m experiments.evaluate_pg19 \
  --model meta-llama/Llama-3.1-8B \
  --methods popkv \
  --context-length 32768 \
  --budget 4096 \
  --group-size 128 \
  --dim-group-size 16 \
  --dim-kernel g16 \
  --generate-tokens 128 \
  --num-blocks 10 \
  --seed 42 \
  --output-dir outputs/pg19_32k/popkv_quality
```

Run PopKV-S:

```bash
python -m experiments.evaluate_pg19 \
  --model meta-llama/Llama-3.1-8B \
  --methods popkv \
  --context-length 32768 \
  --budget 4096 \
  --group-size 64 \
  --dim-group-size 64 \
  --dim-kernel g64-u32 \
  --generate-tokens 128 \
  --num-blocks 10 \
  --seed 42 \
  --output-dir outputs/pg19_32k/popkv_speed
```

The PG19 loader downloads the official test-file manifest from Hugging Face and caches the public-domain books under `data/`. Use `--text-file PATH` for a local corpus.

## LongBench evaluation

The included evaluator supports the paper's six-task pilot and resumable output:

```bash
python -m experiments.evaluate_longbench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --methods full,quest,fier,popkv \
  --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa,gov_report,triviaqa \
  --budgets 512,1024,2048,4096 \
  --samples-per-dataset 12 \
  --max-context 32768 \
  --group-size 32 \
  --hard-limit-minutes 240 \
  --stop-reserve-minutes 10 \
  --output-dir outputs/longbench_n12
```

Compact paper tables are retained in [`results/longbench_summary.csv`](results/longbench_summary.csv) and [`results/longbench_per_task.csv`](results/longbench_per_task.csv). The pilot is small and exhibits non-monotonic budget effects; sparse scores above Full attention should be interpreted as competitive quality within the sample, not as evidence that retrieval improves the base model.

## Ablations and microbenchmarks

Token-group sweep:

```bash
python -m experiments.sweep_token_groups --dry-run
python -m experiments.sweep_token_groups \
  --context-length 32768 \
  --budget 4096 \
  --generate-tokens 500 \
  --num-blocks 2 \
  --tok-g-start 2 \
  --tok-g-stop 500 \
  --tok-g-step 2 \
  --output-dir outputs/token_group_sweep
```

CUDA microbenchmarks:

```bash
python -m experiments.benchmarks.benchmark_encoding --dry-run
python -m experiments.benchmarks.benchmark_dimension_groups --dry-run
```

Plot utilities are under `experiments/plots/` and accept the generated CSV/JSON outputs through their command-line arguments.

## Scope and limitations

- This is a research implementation for Llama-style autoregressive decode, not a drop-in replacement for every Transformers architecture.
- Prefix prefill remains dense; the experiments target decode after a long prefix.
- The first two transformer layers use Full attention in sparse configurations by default.
- Selected K/V values stay in the model's original precision. PopKV compresses retrieval metadata and approximate scoring, not the final attention values.
- PG19 PPL is sampled next-token decode PPL from fixed long-context blocks, not conventional full-corpus PPL.
- Absolute latency depends on GPU, CUDA, PyTorch, and compilation settings. Compare methods with identical settings and hardware.

## Paper and authors

- Paper: [PopKV](https://openreview.net/forum?id=7kA636r4L1)
- Authors: [Minseok Choi](https://openreview.net/profile?id=~Minseok_Choi6), [Geonhee Jang](https://openreview.net/profile?id=~Geonhee_Jang1)

Please use the citation metadata provided on the OpenReview paper page.
