import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LdaBoost.algorithm import LdaBoost


def make_correlated_gaussian_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced multiclass dataset with correlated Gaussian predictors."""
    rng = np.random.default_rng(seed)

    covariance = np.full((n_features, n_features), rho, dtype=float)
    np.fill_diagonal(covariance, 1.0)

    n_per_class = n_samples // n_classes
    leftovers = n_samples % n_classes

    parts_x = []
    parts_y = []
    for c in range(n_classes):
        n_c = n_per_class + (1 if c < leftovers else 0)
        mean = np.full(n_features, float(c))
        x_c = rng.multivariate_normal(mean=mean, cov=covariance, size=n_c)
        y_c = np.full(n_c, c, dtype=int)
        parts_x.append(x_c)
        parts_y.append(y_c)

    x = np.vstack(parts_x)
    y = np.concatenate(parts_y)

    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


def fit_predict_gbm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int):
    clf = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=120,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.6,
        random_state=seed,
    )

    t0 = perf_counter()
    clf.fit(x_train, y_train)
    fit_s = perf_counter() - t0

    t1 = perf_counter()
    y_pred = clf.predict(x_test)
    pred_s = perf_counter() - t1

    return y_pred, fit_s, pred_s


def fit_predict_pca_gbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_features: int,
    seed: int,
):
    n_components = min(50, max(2, n_features // 4))
    pca = PCA(n_components=n_components, random_state=seed)

    clf = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=120,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.6,
        random_state=seed,
    )

    t0 = perf_counter()
    x_train_pca = pca.fit_transform(x_train)
    clf.fit(x_train_pca, y_train)
    fit_s = perf_counter() - t0

    t1 = perf_counter()
    x_test_pca = pca.transform(x_test)
    y_pred = clf.predict(x_test_pca)
    pred_s = perf_counter() - t1

    return y_pred, fit_s, pred_s


def fit_predict_lda_gbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    seed: int,
):
    lda = LinearDiscriminantAnalysis(n_components=min(n_classes - 1, x_train.shape[1]))

    clf = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=120,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.6,
        random_state=seed,
    )

    t0 = perf_counter()
    x_train_lda = lda.fit_transform(x_train, y_train)
    clf.fit(x_train_lda, y_train)
    fit_s = perf_counter() - t0

    t1 = perf_counter()
    x_test_lda = lda.transform(x_test)
    y_pred = clf.predict(x_test_lda)
    pred_s = perf_counter() - t1

    return y_pred, fit_s, pred_s


def fit_predict_ldaboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
):
    model = LdaBoost(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )

    t0 = perf_counter()
    model.fit(x_train, y_train)
    fit_s = perf_counter() - t0

    t1 = perf_counter()
    y_pred = model.predict(x_test)
    pred_s = perf_counter() - t1

    return y_pred, fit_s, pred_s


def run_runtime_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = [
        {
            "scenario": "binary_high_dim",
            "n_samples": 1200,
            "n_features": 300,
            "n_classes": 2,
            "rho": 0.5,
        },
        {
            "scenario": "multiclass_moderate_dim",
            "n_samples": 1200,
            "n_features": 100,
            "n_classes": 5,
            "rho": 0.5,
        },
    ]
    seeds = [11, 29, 47, 71, 89]

    records = []

    for cfg in scenarios:
        for seed in seeds:
            x, y = make_correlated_gaussian_data(
                n_samples=cfg["n_samples"],
                n_features=cfg["n_features"],
                n_classes=cfg["n_classes"],
                rho=cfg["rho"],
                seed=seed,
            )

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.25,
                stratify=y,
                random_state=seed,
            )

            methods = {
                "GBM": lambda: fit_predict_gbm(x_train, y_train, x_test, seed),
                "PCA+GBM": lambda: fit_predict_pca_gbm(
                    x_train, y_train, x_test, cfg["n_features"], seed
                ),
                "LDA+GBM": lambda: fit_predict_lda_gbm(
                    x_train, y_train, x_test, cfg["n_classes"], seed
                ),
                "LdaBoost": lambda: fit_predict_ldaboost(x_train, y_train, x_test, seed),
            }

            for model_name, run_method in methods.items():
                y_pred, fit_s, pred_s = run_method()
                records.append(
                    {
                        "scenario": cfg["scenario"],
                        "seed": seed,
                        "model": model_name,
                        "fit_seconds": fit_s,
                        "predict_seconds": pred_s,
                        "total_seconds": fit_s + pred_s,
                        "test_accuracy": accuracy_score(y_test, y_pred),
                    }
                )

    raw = pd.DataFrame.from_records(records)

    summary = (
        raw.groupby(["scenario", "model"], as_index=False)
        .agg(
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", "std"),
            total_seconds_mean=("total_seconds", "mean"),
            total_seconds_std=("total_seconds", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
        )
        .sort_values(["scenario", "model"])
    )

    return raw, summary


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output_pipeline_confront"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, summary_df = run_runtime_ablation()

    raw_csv = output_dir / "runtime_ablation_raw.csv"
    summary_csv = output_dir / "runtime_ablation_summary.csv"
    summary_json = output_dir / "runtime_ablation_summary.json"

    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "rows": summary_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved raw results to: {raw_csv}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved summary JSON to: {summary_json}")


if __name__ == "__main__":
    main()
