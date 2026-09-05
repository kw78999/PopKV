# Split versus serial indexed attention

Old rows come from the immediately preceding 10-block serial-attention runs. Full is unchanged by split attention.

| Method | Gather+attention serial | Split | Change | Generation serial | Split | Change | PPL delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 143.666 | 143.341 | -0.23% | 171.051 | 170.742 | -0.18% | +0.0000 |
| QUEST | 26.694 | 12.385 | -53.60% | 56.250 | 42.157 | -25.05% | +0.0085 |
| FIER | 25.304 | 12.521 | -50.52% | 57.854 | 45.581 | -21.21% | +0.0019 |
| PopKV-Q | 25.326 | 12.447 | -50.85% | 56.332 | 43.888 | -22.09% | -0.0007 |
| PopKV-S | 25.389 | 12.450 | -50.96% | 55.662 | 43.011 | -22.73% | +0.0104 |
