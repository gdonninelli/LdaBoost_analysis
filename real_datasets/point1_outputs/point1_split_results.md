# Point 1 Split Results (Baseline Commit)

This report tracks Point 1 retuning results for different inner/outer split configurations.

## Run R1 (completed)

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

### Accuracy summary (mean +/- std over outer folds)

| Dataset | PCA+GBM | LDA+GBM |
|---|---:|---:|
| HAR | 0.9456 +/- 0.0094 | 0.9791 +/- 0.0036 |
| RAINFALL | 0.8626 +/- 0.0152 | 0.8603 +/- 0.0207 |
| IRIS | 0.9667 +/- 0.0306 | 0.9800 +/- 0.0200 |
| SONAR | 0.7645 +/- 0.0158 | 0.6878 +/- 0.0518 |
| YEAST | 0.5822 +/- 0.0232 | 0.6139 +/- 0.0296 |

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
