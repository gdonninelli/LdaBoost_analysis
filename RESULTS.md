- Outer splits: 3
- Inner splits: 3
- Methods: GBM, PCA+GBM, LDA+GBM, LdaBoost
- Scope: learning-rate-only
- Learning-rate grid: 0.03, 0.05, 0.07, 0.10
- Seed: 42
- n_jobs: 8

### Cross-validated accuracy table

| Dataset | GBM | PCA+GBM | LDA+GBM | LdaBoost |
|---|---:|---:|---:|---:|
| HAR | **0.995 ± 0.002** | 0.942 ± 0.006 | 0.978 ± 0.004 | 0.978 ± 0.006 |
| RAINFALL | 0.861 ± 0.025 | 0.863 ± 0.015 | 0.860 ± 0.021 | **0.864 ± 0.025** |
| IRIS | 0.960 ± 0.044 | 0.967 ± 0.031 | **0.980 ± 0.020** | 0.967 ± 0.031 |
| SONAR | 0.694 ± 0.173 | **0.764 ± 0.016** | 0.688 ± 0.052 | 0.721 ± 0.075 |
| YEAST | 0.568 ± 0.037 | 0.582 ± 0.023 | **0.614 ± 0.030** | 0.589 ± 0.032 |

### Mean runtime by dataset and method

| Dataset | PCA+GBM | LDA+GBM | LdaBoost | GBM |
|---|---:|---:|---:|---:|
| HAR | 00:06:47 | 00:00:09 | 00:03:24 | 00:07:09 |
| RAINFALL | 00:00:01 | 00:00:01 | 00:00:02 | 00:00:00 |
| IRIS | 00:00:00 | 00:00:00 | 00:00:02 | 00:00:00 |
| SONAR | 00:00:00 | 00:00:00 | 00:00:02 | 00:00:00 |
| YEAST | 00:00:03 | 00:00:03 | 00:00:16 | 00:00:02 |

### Test validation accuracy table

Holdout setup for this block:
- HAR (official split): train size = 7352, test size = 2947.
- RAINFALL (train split fallback): train size = 1752, test size = 438.

| Dataset | GBM | PCA+GBM | LDA+GBM | LdaBoost |
|---|---:|---:|---:|---:|
| HAR | 0.9400 | 0.9053 | **0.9528** | 0.9511 |
| RAINFALL | **0.8790** | 0.8721 | 0.8607 | 0.8584 |

### Test validation timing table

| Method | HAR Runtime (s) | RAINFALL Runtime (s) |
|---|---:|---:|
| GBM | 675.07 | 0.53 |
| LDA+GBM | 10.25 | 0.27 |
| PCA+GBM | 652.62 | 0.70 |
| LdaBoost | 286.22 | 2.33 |

From a computational perspective, the LDA-based approaches remain very competitive while keeping strong accuracy. One-shot LDA+GBM is generally the fastest option (and in HAR it is dramatically faster than PCA+GBM), while still delivering high predictive performance. At the same time, LdaBoost deserves a positive emphasis: despite its iterative LDA updates, it keeps a favorable speed-accuracy trade-off, remains substantially faster than PCA+GBM on massive dataset such as HAR, and preserves consistently strong predictive performance across datasets.
