| Model | Variant | Params | Runs | Test error % (mean ± std) | Each seed | Paper |
|---|---|---|---|---|---|---|
| plain-20 | main (A, AMP) | 0.27M | 3 | 9.56 ± 0.20 | s0: 9.39, s1: 9.50, s2: 9.78 | ≈ 9 to 10 (Fig. 6) |
| plain-32 | main (A, AMP) | 0.46M | 1 | 9.89 (1 run) | s0: 9.89 | n/a |
| plain-44 | main (A, AMP) | 0.66M | 1 | 10.94 (1 run) | s0: 10.94 | n/a |
| plain-56 | main (A, AMP) | 0.85M | 3 | 12.79 ± 0.41 | s0: 12.65, s1: 13.25, s2: 12.46 | ≈ 13 (Fig. 6) |
| ResNet-20 | main (A, AMP) | 0.27M | 3 | 8.36 ± 0.27 | s0: 8.62, s1: 8.09, s2: 8.37 | 8.75 |
| ResNet-32 | main (A, AMP) | 0.46M | 1 | 7.78 (1 run) | s0: 7.78 | 7.51 |
| ResNet-44 | main (A, AMP) | 0.66M | 3 | 7.41 ± 0.30 | s0: 7.53, s1: 7.06, s2: 7.63 | 7.17 |
| ResNet-56 | main (A, AMP) | 0.85M | 3 | 7.93 ± 0.23 | s0: 8.04, s1: 8.08, s2: 7.66 | 6.97 |
| ResNet-110 | main (A, AMP) | 1.73M | 1 | 6.50 (1 run) | s0: 6.50 | 6.43 (6.61 ± 0.16) |
| ResNet-56 | B | 0.86M | 3 | 7.06 ± 0.33 | s0: 6.83, s1: 7.44, s2: 6.92 | not reported |
| ResNet-56 | fp32 | 0.85M | 1 | 7.95 (1 run) | s0: 7.95 | not reported |
