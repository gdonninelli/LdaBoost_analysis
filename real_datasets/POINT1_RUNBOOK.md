# Point 1 Retuning Runbook

This runbook prepares the experiment requested in Point 1:

- independent retuning across methods,
- practical fallback: retune only `learning_rate` around `0.05`,
- keep GBM-based `subsample` fixed at `0.6`.

The runner script is:

- `real_datasets/run_point1_retuning.py`

## 1) Recommended first run (fast, learning-rate only)

```bash
python real_datasets/run_point1_retuning.py \
  --scope learning-rate-only \
  --learning-rates 0.03,0.05,0.07,0.10 \
  --inner-splits 5 \
  --n-jobs -1
```

Notes:

- This keeps tree parameters fixed and retunes only `learning_rate`.
- It is the closest implementation of: "if too long, tune only learning_rate around 0.05".

## 2) Full retuning run (heavier)

```bash
python real_datasets/run_point1_retuning.py \
  --scope full \
  --learning-rates 0.03,0.05,0.07,0.10 \
  --full-n-estimators 100,200,300 \
  --full-max-depth 2,3,4 \
  --full-min-samples-leaf 1,5,10 \
  --inner-splits 5 \
  --n-jobs -1
```

## 3) Optional: run a subset of datasets first

```bash
python real_datasets/run_point1_retuning.py \
  --datasets HAR,YEAST \
  --scope learning-rate-only
```

## 4) Output artifacts

By default, outputs are written to:

- `real_datasets/point1_outputs/point1_retune_fold_accuracies.json`
- `real_datasets/point1_outputs/point1_retune_fold_accuracy_summary.csv`
- `real_datasets/point1_outputs/point1_retune_best_params.csv`
- `real_datasets/point1_outputs/point1_retune_run_config.json`

## 5) Important implementation note

`subsample=0.6` is fixed for GBM, PCA+GBM, and LDA+GBM.

The custom `LdaBoost` class in this repository does not expose a `subsample` parameter, so this constraint does not apply to that method unless the class is extended.