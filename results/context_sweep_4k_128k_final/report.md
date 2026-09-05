# PG19 4–128K context sweep: consolidated results

Configuration: Llama-3.1-8B, budget 4096, 4 PG19 blocks, 256 evaluated tokens per block, seed 42.

Source policy: 4–31K uses the new low-context multi-method sweep. From 32K onward, Quest uses the latest GQA-fused sweep while Full, FIER, PopKV-Q, and PopKV-S use the earlier multi-method sweep. Results with fewer than 1024 evaluated tokens are excluded. One-block experiments are not mixed into this table.

Coverage: Full 37/125, Quest (GQA fused) 125/125, FIER 122/125, PopKV-Q 122/125, PopKV-S 122/125.

| Context | Method | PPL ↓ | Decode ms/token ↓ | Tokens/s ↑ |
|---:|:---|---:|---:|---:|
| 4K | Full | 6.90 | 44.34 | 22.05 |
| 4K | Quest (GQA fused) | 7.01 | 47.25 | 20.34 |
| 4K | FIER | 6.89 | 44.17 | 21.71 |
| 4K | PopKV-Q | 6.89 | 44.96 | 21.22 |
| 4K | PopKV-S | 6.88 | 42.90 | 22.30 |
| 8K | Full | 5.24 | 60.13 | 16.35 |
| 8K | Quest (GQA fused) | 5.82 | 43.35 | 22.13 |
| 8K | FIER | 5.24 | 42.91 | 22.33 |
| 8K | PopKV-Q | 5.23 | 45.14 | 21.11 |
| 8K | PopKV-S | 5.23 | 43.01 | 22.22 |
| 16K | Full | 9.40 | 95.85 | 10.33 |
| 16K | Quest (GQA fused) | 11.08 | 38.91 | 24.66 |
| 16K | FIER | 9.46 | 40.18 | 23.89 |
| 16K | PopKV-Q | 9.47 | 43.79 | 21.77 |
| 16K | PopKV-S | 9.46 | 40.93 | 23.39 |
| 32K | Full | 11.86 | 194.85 | 5.11 |
| 32K | Quest (GQA fused) | 14.55 | 47.51 | 20.36 |
| 32K | FIER | 11.90 | 49.96 | 19.46 |
| 32K | PopKV-Q | 11.96 | 47.99 | 20.11 |
| 32K | PopKV-S | 12.00 | 47.44 | 20.41 |
| 64K | Full | — | — | — |
| 64K | Quest (GQA fused) | 14.71 | 71.39 | 13.70 |
| 64K | FIER | 12.35 | 77.06 | 12.71 |
| 64K | PopKV-Q | 12.42 | 73.22 | 13.30 |
| 64K | PopKV-S | 12.95 | 72.47 | 13.46 |
| 96K | Full | — | — | — |
| 96K | Quest (GQA fused) | 18.83 | 90.33 | 10.86 |
| 96K | FIER | 10.61 | 115.15 | 8.57 |
| 96K | PopKV-Q | 10.80 | 109.06 | 9.01 |
| 96K | PopKV-S | 12.13 | 107.99 | 9.11 |
| 125K | Full | — | — | — |
| 125K | Quest (GQA fused) | 20.80 | 118.10 | 8.33 |
| 125K | FIER | 10.27 | 148.41 | 6.68 |
| 125K | PopKV-Q | 10.47 | 140.18 | 7.04 |
| 125K | PopKV-S | 11.77 | 138.91 | 7.12 |
| 128K | Full | — | — | — |
| 128K | Quest (GQA fused) | 25.65 | 120.79 | 8.16 |
| 128K | FIER | — | — | — |
| 128K | PopKV-Q | — | — | — |
| 128K | PopKV-S | — | — | — |

`—` means that no complete, configuration-compatible measurement is available.

Files: `final_context_table.csv` is the one-row-per-context final table; `combined_long.csv` preserves all available metrics and source provenance; `checkpoint_summary.csv` contains the compact checkpoints shown above.
