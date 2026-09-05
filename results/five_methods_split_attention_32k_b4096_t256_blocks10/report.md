# PG19 32K: quality and token-generation time

Llama-3.1-8B; context 32,768; budget 4,096; 10 blocks × 256 evaluated tokens; BF16; seed 42. Layers 0 and 1 remain dense.

## Quality and overall latency

| Method | PPL | Generation ms/token | token/s |
|---|---:|---:|---:|
| Full | 11.1416 | 170.742 | 5.857 |
| QUEST | 13.8755 | 42.157 | 23.721 |
| FIER | 11.1880 | 45.581 | 21.939 |
| PopKV-Q | 11.2354 | 43.888 | 22.785 |
| PopKV-S | 11.2626 | 43.011 | 23.250 |

## Additive token-generation breakdown

Each component is exclusive. Total = score + Top-k + attention + other model operations + decode timing remainder + next-token selection + cache update.

| Stage (ms/token) | Full | QUEST | FIER | PopKV-Q | PopKV-S |
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

Score includes query preparation and approximate scoring, excluding Top-k. The historical raw data records this as one score interval; newer evaluations record query encoding and scoring separately, which this script adds. A separate historical query-encoding time cannot be recovered.

Attention includes dense layers and sparse partial/merge kernels. Other model operations include frontend, QKV/RoPE, full K/V concatenation, output projection, MLP, and final norm/head. The timing remainder retains measured GPU intervals outside named stages and host residual time. Next-token selection chooses the output token; Top-k chooses K/V positions.

The display rounding adjustment only reconciles independently rounded numbers (0.001 ms precision); it is not a measured stage. CSV values retain full precision. Closure is checked within 1e-6 ms/token before writing tables.

Prefix prefill and one-time metadata prepack are outside this decode-cycle total. Prepack measurements are preserved in overall.csv (ms/block); token/s = 1000 / total token-generation ms.

## Decode stages (including timing remainders)

| Stage (ms/token) | Full | QUEST | FIER | PopKV-Q | PopKV-S |
|---|---:|---:|---:|---:|---:|
| Frontend | 0.383 | 0.359 | 0.361 | 0.355 | 0.356 |
| QKV + RoPE | 4.675 | 4.561 | 4.580 | 4.552 | 4.533 |
| Score (query preparation + scoring) | 0.000 | 1.006 | 3.227 | 1.365 | 0.813 |
| Top-k retrieval | 0.000 | 0.882 | 1.844 | 1.814 | 1.769 |
| Full K/V concatenation | 19.221 | 1.164 | 1.172 | 1.162 | 1.162 |
| Full attention | 124.120 | 7.551 | 7.588 | 7.542 | 7.545 |
| Sparse attention partial | 0.000 | 3.526 | 3.618 | 3.600 | 3.602 |
| Sparse attention merge | 0.000 | 0.144 | 0.143 | 0.143 | 0.141 |
| Output projection | 1.852 | 1.856 | 1.873 | 1.869 | 1.851 |
| MLP | 17.782 | 17.701 | 17.735 | 17.717 | 17.664 |
| Final norm + LM head | 1.511 | 1.509 | 1.510 | 1.510 | 1.509 |
| GPU timing remainder | 0.009 | 0.009 | 0.009 | 0.009 | 0.009 |
| Host decode remainder | 0.429 | 0.479 | 0.488 | 0.473 | 0.478 |
| Display rounding adjustment | 0.000 | -0.001 | -0.001 | 0.000 | 0.000 |
| **Decode total** | **169.982** | **40.746** | **44.147** | **42.111** | **41.432** |

## Cache-update stages

| Stage (ms/token) | Full | QUEST | FIER | PopKV-Q | PopKV-S |
|---|---:|---:|---:|---:|---:|
| Static KV update | 0.649 | 0.650 | 0.660 | 0.642 | 0.650 |
| Active-key staging | 0.000 | 0.402 | 0.410 | 0.414 | 0.417 |
| K metadata encoding | 0.000 | 0.184 | 0.189 | 0.599 | 0.385 |
| Metadata write | 0.000 | 0.010 | 0.010 | 0.003 | 0.004 |
| Group sealing | 0.000 | 0.022 | 0.020 | 0.004 | 0.006 |
| Retrieval GPU remainder | 0.000 | 0.050 | 0.051 | 0.028 | 0.031 |
| Update host remainder | 0.042 | 0.031 | 0.032 | 0.026 | 0.027 |
| Display rounding adjustment | 0.000 | 0.001 | 0.001 | 0.001 | -0.001 |
| **Update total** | **0.691** | **1.350** | **1.373** | **1.717** | **1.519** |

The decode and update tables expand components of the generation table; do not add their totals a second time. Numerical checks are in latency_closure.csv.
