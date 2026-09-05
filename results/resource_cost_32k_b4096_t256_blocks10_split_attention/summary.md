# Resource cost summary (10 blocks, 256 generated tokens/block)

Values are means per block. Analytical counts are identical for equal shapes.

| Method | TFLOP | Compare (G) | Integer (G) | Bitwise (G) | POpcount (G) | Convert (G) | Exp (M) | rsqrt (M) | Top-k inputs (G) | Logical traffic (TB) | Metadata (MiB) | Total cache / Full KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 8.278 | 8.656 | 0.000 | 0.000 | 0.000 | 0.000 | 8740.798 | 0.017 | 0.000 | 10.540 | 0.00 | 1.000x |
| QUEST | 4.834 | 72.302 | 4.027 | 0.000 | 0.000 | 5.131 | 1666.966 | 0.017 | 0.505 | 5.349 | 483.75 | 1.117x |
| FIER-author | 8.782 | 6.926 | 1.982 | 2071.816 | 0.000 | 1144.406 | 1666.966 | 0.017 | 8.084 | 5.176 | 241.88 | 1.059x |
| PopKV-Q | 5.622 | 9.744 | 518.393 | 585.150 | 517.386 | 302.204 | 1666.966 | 0.262 | 8.084 | 4.982 | 483.75 | 1.117x |
| PopKV-S | 4.894 | 8.722 | 130.353 | 293.617 | 226.356 | 83.429 | 1666.966 | 0.262 | 8.084 | 4.930 | 302.34 | 1.073x |

FMA=2 FLOPs. Comparisons, integer arithmetic, bitwise operations, 32-bit CUDA popcount, conversions, exp, and rsqrt remain separate. topk_input_elements reports the exact selection workload cardinality because the internal CUDA top-k comparator/integer instruction count is backend- and data-dependent.

Logical traffic is operator-level tensor read+write traffic.

All sparse methods retain the BF16 K/V cache. Metadata is therefore additional storage; the final column reports total cache storage relative to Full K/V.
