# Point 1 Split Results (Baseline Commit)

This report tracks Point 1 retuning results for different inner/outer split configurations.

## Run R1 (completed)

Due to the high computational burden of nested grid-search tuning (in particular on HAR), this phase was run with a reduced nested validation setup: 3 inner folds and 3 outer folds. This preserves a cross-validated comparison while keeping runtime manageable.

- Outer splits: 3
- Inner splits: 3
- Methods: PCA+GBM, LDA+GBM
- Scope: learning-rate-only
- Learning-rate grid: 0.03, 0.05, 0.07, 0.10
- Seed: 42
- n_jobs: 8
- Total elapsed: 00:31:06

Source files:
- `point1_retune_run_config.json`
- `point1_retune_fold_accuracy_summary.csv`
- `point1_retune_best_params.csv`
- `point1_retune_fold_timing.csv`

### Accuracy summary and comparison with manuscript baselines

R1 values are from this run (outer=3, inner=3). GBM and LdaBoost reference values are from Table 1 in `Paper/main.tex` (cross-validated accuracy).

| Dataset | GBM (main.tex) | PCA+GBM (R1) | LDA+GBM (R1) | LdaBoost (main.tex) |
|---|---:|---:|---:|---:|
| HAR | 0.995 | 0.9456 +/- 0.0094 | 0.9791 +/- 0.0036 | 0.978 |
| RAINFALL | 0.861 | 0.8626 +/- 0.0152 | 0.8603 +/- 0.0207 | 0.864 |
| IRIS | 0.960 | 0.9667 +/- 0.0306 | 0.9800 +/- 0.0200 | 0.967 |
| SONAR | 0.694 | 0.7645 +/- 0.0158 | 0.6878 +/- 0.0518 | 0.721 |
| YEAST | 0.568 | 0.5822 +/- 0.0232 | 0.6139 +/- 0.0296 | 0.589 |

### Most frequent selected learning rate (across folds)

| Dataset | PCA+GBM | LDA+GBM |
|---|---:|---:|
| HAR | 0.10 | 0.03 |
| RAINFALL | 0.03 | 0.03 |
| IRIS | 0.03 | 0.03 |
| SONAR | 0.07 | 0.03 |
| YEAST | 0.03 | 0.03 |

### Mean fold runtime by dataset (seconds)

| Dataset | Mean fold time (s) |
|---|---:|
| HAR | 614.21 |
| RAINFALL | 0.82 |
| IRIS | 0.63 |
| SONAR | 0.40 |
| YEAST | 5.65 |

## Split-comparison tracker

Add future runs below to compare split settings:

| Run ID | Outer splits | Inner splits | Methods | Total elapsed | Notes |
|---|---:|---:|---|---|---|
| R1 | 3 | 3 | PCA+GBM, LDA+GBM | 00:31:06 | Baseline commit |
| R2 | 3 | 3 | PCA+GBM, LDA+GBM | 14:32:36 | Full grid-search completed |
| R3 | 3 | 3 | LDA+GBM, PCA+GBM, LdaBoost | checkpoint at dataset completion | Cross-validation completed on all datasets; holdout validation pending |

## Run R2 (completed)

- Outer splits: 3
- Inner splits: 3
- Methods: PCA+GBM, LDA+GBM
- Scope: full
- Learning-rate grid: 0.03, 0.05, 0.07, 0.10
- n_estimators grid: 100, 200, 300
- max_depth grid: 2, 3, 4
- min_samples_leaf grid: 1, 5, 10
- Seed: 42
- n_jobs: 8
- Total elapsed: 14:32:36

Command used/planned:

```bash
cd "/Users/giuliodonninelli/Documents/04 Università/01 Statistica UNIMIB/LdaBoost_analysis" && /Users/giuliodonninelli/.local/share/virtualenvs/remarkable-substack-D7zpFFy-/bin/python real_datasets/run_point1_retuning.py --methods PCA+GBM,LDA+GBM --scope full --outer-splits 3 --inner-splits 3 --learning-rates 0.03,0.05,0.07,0.10 --full-n-estimators 100,200,300 --full-max-depth 2,3,4 --full-min-samples-leaf 1,5,10 --n-jobs 8
```

### Accuracy summary and comparison with manuscript baselines

| Dataset | GBM (main.tex) | PCA+GBM (R2) | LDA+GBM (R2) | LdaBoost (main.tex) |
|---|---:|---:|---:|---:|
| HAR | 0.995 | 0.9678 +/- 0.0075 | 0.9791 +/- 0.0051 | 0.978 |
| RAINFALL | 0.861 | 0.8616 +/- 0.0202 | 0.8630 +/- 0.0151 | 0.864 |
| IRIS | 0.960 | 0.9800 +/- 0.0200 | 0.9800 +/- 0.0200 | 0.967 |
| SONAR | 0.694 | 0.7885 +/- 0.0076 | 0.6879 +/- 0.0700 | 0.721 |
| YEAST | 0.568 | 0.5815 +/- 0.0226 | 0.6112 +/- 0.0348 | 0.589 |

### Most frequent selected hyperparameters in R2 (across folds)

| Dataset | Method | n_estimators | max_depth | min_samples_leaf | learning_rate |
|---|---|---:|---:|---:|---:|
| HAR | PCA+GBM | 300 | 4 | 5 | 0.10 |
| HAR | LDA+GBM | 100 | 4 | 5 | 0.07 |
| RAINFALL | PCA+GBM | 100 | 2 | 5 | 0.03 |
| RAINFALL | LDA+GBM | 100 | 2 | 5 | 0.03 |
| IRIS | PCA+GBM | 300 | 2 | 10 | 0.05 |
| IRIS | LDA+GBM | 100 | 2 | 1 | 0.03 |
| SONAR | PCA+GBM | 200 | 4 | 10 | 0.10 |
| SONAR | LDA+GBM | 100 | 2 | 1 | 0.03 |
| YEAST | PCA+GBM | 100 | 4 | 5 | 0.03 |
| YEAST | LDA+GBM | 100 | 2 | 1 | 0.03 |

### Mean fold runtime by dataset in R2

| Dataset | Mean fold time |
|---|---:|
| HAR | 04:46:57 |
| RAINFALL | 00:00:24 |
| IRIS | 00:00:17 |
| SONAR | 00:00:10 |
| YEAST | 00:03:04 |

----
# Run R3 (cross-validation completed, holdout pending)

- Outer splits: 3
- Inner splits: 3
- Methods: LDA+GBM, PCA+GBM, LdaBoost
- Scope: learning-rate-only
- Learning-rate grid: 0.03, 0.05, 0.07, 0.10
- Seed: 42
- n_jobs: 8
- Run stage from config: `dataset_completed:YEAST` (all CV datasets completed; final holdout block not completed)

Command used/planned:

```bash
cd "/Users/giuliodonninelli/Documents/04 Università/01 Statistica UNIMIB/LdaBoost_analysis" && /Users/giuliodonninelli/.local/share/virtualenvs/remarkable-substack-D7zpFFy-/bin/python real_datasets/run_point1_retuning.py --methods "LDA+GBM,PCA+GBM,LdaBoost" --scope learning-rate-only --outer-splits 3 --inner-splits 3 --learning-rates 0.03,0.05,0.07,0.10 --n-jobs 8
```

Source files used for R3 status:
- `point1_retune_run_config.json`
- `point1_retune_fold_accuracy_summary.csv`
- `point1_retune_best_params.csv`
- `point1_retune_fold_timing.csv`
- `point1_retune_holdout_test_accuracy.csv` (currently empty)

### Cross-validated accuracy table (R3)

| Dataset | GBM | LDA+GBM | PCA+GBM | LdaBoost |
|---|---:|---:|---:|---:|
| HAR | 0.995 ± 0.002 | 0.978 ± 0.004 | 0.942 ± 0.006 | 0.978 ± 0.006 |
| RAINFALL | 0.861 ± 0.025 | 0.860 ± 0.021 | 0.863 ± 0.015 | 0.864 ± 0.025 |
| IRIS | 0.960 ± 0.044 | 0.980 ± 0.020 | 0.967 ± 0.031 | 0.967 ± 0.031 |
| SONAR | 0.694 ± 0.173 | 0.688 ± 0.052 | 0.764 ± 0.016 | 0.721 ± 0.075 |
| YEAST | 0.568 ± 0.037 | 0.614 ± 0.030 | 0.582 ± 0.023 | 0.589 ± 0.032 |

### Mean runtime by dataset and method (R3, per outer fold)

| Dataset | LDA+GBM | PCA+GBM | LdaBoost | GBM |
|---|---:|---:|---:|---:|
| HAR | 00:00:09 | 00:06:47 | 00:03:24 | 00:07:09 |
| RAINFALL | 00:00:01 | 00:00:01 | 00:00:02 | 00:00:00 |
| IRIS | 00:00:00 | 00:00:00 | 00:00:02 | 00:00:00 |
| SONAR | 00:00:00 | 00:00:00 | 00:00:02 | 00:00:00 |
| YEAST | 00:00:03 | 00:00:03 | 00:00:16 | 00:00:02 |

### Test validation accuracy table (R3, HAR and RAINFALL holdout)

Holdout setup for this block:
- HAR (official split): train size = 7352, test size = 2947, inner_splits = 3.
- RAINFALL (train split fallback): train size = 1752, test size = 438, inner_splits = 3.

| Dataset | GBM | LDA+GBM | PCA+GBM | LdaBoost |
|---|---:|---:|---:|---:|
| HAR | 0.9400 | 0.9528 | 0.9053 | 0.9511 |
| RAINFALL | 0.8790 | 0.8607 | 0.8721 | 0.8584 |

### Test validation timing table (R3)

| Method | HAR Runtime (s) | RAINFALL Runtime (s) |
|---|---:|---:|
| GBM | 675.07 | 0.53 |
| LDA+GBM | 10.25 | 0.27 |
| PCA+GBM | 652.62 | 0.70 |
| LdaBoost | 286.22 | 2.33 |

From a computational perspective, the LDA-based approaches remain very competitive while keeping strong accuracy. One-shot LDA+GBM is generally the fastest option (and in HAR it is dramatically faster than PCA+GBM), while still delivering high predictive performance. At the same time, LdaBoost deserves a positive emphasis: despite its iterative LDA updates, it keeps a favorable speed-accuracy trade-off, remains substantially faster than PCA+GBM on massive dataset such as HAR, and preserves consistently strong predictive performance across datasets.
