# LongBench Pop-KV reproduction

Scores are percentages; only completed samples are included. Average is the unweighted mean of task scores. Retention is the mean of per-task score / Full score ratios, not the ratio of task averages. See experiment_config.json for the model and context settings.

## Task scores

| Method | Budget | qasper | multifieldqa_en | hotpotqa | gov_report | triviaqa | Average |
|---|---|---|---|---|---|---|---|
| fier | 4096 | 46.146 | 53.630 | 56.058 | 19.972 | 91.478 | 53.457 |
| full |  | 45.982 | 53.050 | 55.354 | 19.917 | 91.645 | 53.190 |
| popkv-q | 4096 | 46.042 | 53.553 | 55.258 | 19.931 | 91.478 | 53.252 |
| popkv-s | 4096 | 45.602 | 53.929 | 54.891 | 20.042 | 91.469 | 53.187 |
| quest | 4096 | 47.277 | 53.550 | 55.369 | 20.111 | 89.669 | 53.195 |

## Retention vs Full

| Method | Budget | Average | Retention % | Min task % |
|---|---|---|---|---|
| fier | 4096 | 53.457 | 100.563 | 99.818 |
| full |  | 53.190 | 100.000 | 100.000 |
| popkv-q | 4096 | 53.252 | 100.158 | 99.818 |
| popkv-s | 4096 | 53.187 | 100.086 | 99.165 |
| quest | 4096 | 53.195 | 100.520 | 97.844 |

## Runtime

| Method | Budget | Dataset | Decode ms/tok | Search | Attention | Failed |
|---|---|---|---|---|---|---|
| fier | 4096 | gov_report | 40.191 | 2.828 | 6.381 | 0 |
| fier | 4096 | hotpotqa | 45.518 | 3.880 | 7.899 | 0 |
| fier | 4096 | multifieldqa_en | 44.511 | 3.016 | 6.795 | 0 |
| fier | 4096 | qasper | 43.237 | 2.488 | 6.221 | 0 |
| fier | 4096 | triviaqa | 43.781 | 3.402 | 7.358 | 0 |
| full |  | gov_report | 70.589 | 0.000 | 35.764 | 0 |
| full |  | hotpotqa | 81.396 | 0.000 | 44.781 | 0 |
| full |  | multifieldqa_en | 56.556 | 0.000 | 23.454 | 0 |
| full |  | qasper | 49.055 | 0.000 | 17.310 | 0 |
| full |  | triviaqa | 77.320 | 0.000 | 41.419 | 0 |
| popkv-q | 4096 | gov_report | 42.080 | 2.990 | 6.881 | 0 |
| popkv-q | 4096 | hotpotqa | 46.984 | 3.932 | 8.690 | 0 |
| popkv-q | 4096 | multifieldqa_en | 46.625 | 3.633 | 7.643 | 0 |
| popkv-q | 4096 | qasper | 45.458 | 3.206 | 7.010 | 0 |
| popkv-q | 4096 | triviaqa | 45.519 | 3.522 | 8.062 | 0 |
| popkv-s | 4096 | gov_report | 41.361 | 2.716 | 6.863 | 0 |
| popkv-s | 4096 | hotpotqa | 46.788 | 3.786 | 8.804 | 0 |
| popkv-s | 4096 | multifieldqa_en | 46.006 | 3.455 | 7.526 | 0 |
| popkv-s | 4096 | qasper | 45.135 | 2.977 | 7.213 | 0 |
| popkv-s | 4096 | triviaqa | 45.274 | 3.362 | 8.194 | 0 |
| quest | 4096 | gov_report | 41.112 | 2.302 | 6.938 | 0 |
| quest | 4096 | hotpotqa | 45.871 | 3.143 | 8.606 | 0 |
| quest | 4096 | multifieldqa_en | 47.255 | 4.490 | 7.909 | 0 |
| quest | 4096 | qasper | 46.903 | 5.018 | 7.234 | 0 |
| quest | 4096 | triviaqa | 44.672 | 3.321 | 8.020 | 0 |
