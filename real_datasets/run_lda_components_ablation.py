#!/usr/bin/env python3
"""Run LDA-components ablation for HAR and YEAST.

This script evaluates LDA+GBM while varying the number of retained
discriminant components (r) to support the reviewer-requested sensitivity
analysis:
- HAR (C=6): r in {1, 3, 5}
- YEAST (C=10): r in {1, 5, 9}

Outputs are written as CSV/JSON files under a dedicated output directory.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "real_datasets" / "point1_outputs" / "lda_components_ablation"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    cv_splits: int


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "HAR": DatasetConfig(name="HAR", cv_splits=10),
    "YEAST": DatasetConfig(name="YEAST", cv_splits=5),
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def ensure_numeric_features(df: pd.DataFrame) -> np.ndarray:
    x = df.copy()
    for col in x.columns:
        if x[col].dtype == object:
            x[col] = pd.to_numeric(x[col], errors="raise")
    return x.to_numpy(dtype=float)


def load_har_train_test(base: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(base / "har" / "Data" / "train.csv")
    test = pd.read_csv(base / "har" / "Data" / "test.csv")

    y_train = train["Activity"].astype(str).to_numpy()
    x_train = ensure_numeric_features(train.drop(columns=["Activity"]))
    y_test = test["Activity"].astype(str).to_numpy()
    x_test = ensure_numeric_features(test.drop(columns=["Activity"]))
    return x_train, y_train, x_test, y_test


def load_har_train_only(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_train, y_train, _, _ = load_har_train_test(base)
    return x_train, y_train


def load_yeast(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(base / "yeast" / "Data" / "yeast.csv")
    y = df["name"].astype(str).to_numpy()
    x = ensure_numeric_features(df.drop(columns=["name"]))
    return x, y


def safe_n_splits(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    max_feasible = int(np.min(counts))
    return max(2, min(requested, max_feasible))


def validate_components(requested: List[int], y: np.ndarray, dataset_name: str) -> List[int]:
    max_components = len(np.unique(y)) - 1
    valid = sorted({r for r in requested if 1 <= r <= max_components})
    if not valid:
        raise ValueError(
            f"No valid component values for {dataset_name}. "
            f"Requested={requested}, max allowed={max_components}."
        )
    dropped = sorted(set(requested) - set(valid))
    if dropped:
        print(
            f"[{dataset_name}] ignoring invalid component values: {dropped}. "
            f"Allowed range is [1, {max_components}]."
        )
    return valid


def make_lda_gbm_pipeline(
    n_components: int,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(n_components=n_components)),
            (
                "gb",
                GradientBoostingClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    subsample=0.6,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_cv(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    components_grid: List[int],
    cv_splits: int,
    args,
) -> List[dict]:
    rows: List[dict] = []
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=args.seed)

    for n_components in components_grid:
        print(f"[{dataset_name}] CV for r={n_components} | {cv_splits} folds")
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            fold_start = time.perf_counter()

            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]

            model = make_lda_gbm_pipeline(
                n_components=n_components,
                learning_rate=args.learning_rate,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                seed=args.seed,
            )

            fit_start = time.perf_counter()
            max_lda_components = min(x_train.shape[1], max(1, np.unique(y_train).size - 1))
            if hasattr(model, "named_steps") and "lda" in model.named_steps:
                requested = model.named_steps["lda"].get_params().get("n_components")
                if requested is not None:
                    model.named_steps["lda"].set_params(n_components=min(int(requested), max_lda_components))
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_start

            y_hat = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, y_hat))
            total_seconds = time.perf_counter() - fold_start

            print(
                f"[{dataset_name}] r={n_components} fold {fold_id}/{cv_splits} "
                f"acc={accuracy:.6f} elapsed={format_elapsed(total_seconds)}"
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "n_components": int(n_components),
                    "fold": int(fold_id),
                    "cv_splits": int(cv_splits),
                    "accuracy": accuracy,
                    "fit_seconds": float(fit_seconds),
                    "total_seconds": float(total_seconds),
                }
            )

    return rows


def evaluate_har_holdout(
    base_path: Path,
    components_grid: List[int],
    args,
) -> List[dict]:
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_har_train_test(base_path)

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_raw, y_test_raw], axis=0))
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    rows: List[dict] = []
    for n_components in components_grid:
        start = time.perf_counter()
        model = make_lda_gbm_pipeline(
            n_components=n_components,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )
        fit_start = time.perf_counter()
        model.fit(x_train_raw, y_train)
        fit_seconds = time.perf_counter() - fit_start
        y_hat = model.predict(x_test_raw)
        test_accuracy = float(accuracy_score(y_test, y_hat))
        total_seconds = time.perf_counter() - start

        print(
            f"[HAR holdout] r={n_components} test_acc={test_accuracy:.6f} "
            f"elapsed={format_elapsed(total_seconds)}"
        )

        rows.append(
            {
                "dataset": "HAR",
                "n_components": int(n_components),
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "test_accuracy": test_accuracy,
                "fit_seconds": float(fit_seconds),
                "total_seconds": float(total_seconds),
            }
        )

    return rows


def summarize_cv(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "n_components",
                "accuracy_mean",
                "accuracy_std",
                "fit_seconds_mean",
                "fit_seconds_std",
                "total_seconds_mean",
                "total_seconds_std",
                "n_folds",
            ]
        )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["dataset", "n_components"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", "std"),
            total_seconds_mean=("total_seconds", "mean"),
            total_seconds_std=("total_seconds", "std"),
            n_folds=("fold", "count"),
        )
        .sort_values(["dataset", "n_components"])
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LDA-components ablation for HAR and YEAST using LDA+GBM."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="HAR,YEAST",
        help="Comma-separated datasets. Allowed: HAR,YEAST",
    )
    parser.add_argument(
        "--har-components",
        type=str,
        default="1,3,5",
        help="Comma-separated r values for HAR.",
    )
    parser.add_argument(
        "--yeast-components",
        type=str,
        default="1,5,9",
        help="Comma-separated r values for YEAST.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--har-cv-splits", type=int, default=10)
    parser.add_argument("--yeast-cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for ablation artifacts.",
    )
    parser.add_argument(
        "--skip-har-holdout",
        action="store_true",
        help="Skip HAR official holdout evaluation.",
    )
    args = parser.parse_args()

    selected_datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in selected_datasets if d not in DATASET_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed: {list(DATASET_CONFIGS.keys())}")

    base_path = PROJECT_ROOT / "real_datasets"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    started_at = now_utc_iso()
    print(f"[RUN] start: {started_at}")

    requested_components = {
        "HAR": parse_int_list(args.har_components),
        "YEAST": parse_int_list(args.yeast_components),
    }

    cv_rows: List[dict] = []
    selected_components: Dict[str, List[int]] = {}
    selected_cv_splits: Dict[str, int] = {}

    for dataset_name in selected_datasets:
        if dataset_name == "HAR":
            x_raw, y_raw = load_har_train_only(base_path)
            requested_cv_splits = args.har_cv_splits
        else:
            x_raw, y_raw = load_yeast(base_path)
            requested_cv_splits = args.yeast_cv_splits

        y_enc = LabelEncoder().fit_transform(y_raw)
        comp_grid = validate_components(requested_components[dataset_name], y_enc, dataset_name)
        cv_splits = safe_n_splits(y_enc, requested_cv_splits)

        selected_components[dataset_name] = comp_grid
        selected_cv_splits[dataset_name] = cv_splits

        dataset_rows = evaluate_cv(
            dataset_name=dataset_name,
            x=x_raw,
            y=y_enc,
            components_grid=comp_grid,
            cv_splits=cv_splits,
            args=args,
        )
        cv_rows.extend(dataset_rows)

    holdout_rows: List[dict] = []
    if "HAR" in selected_datasets and not args.skip_har_holdout:
        holdout_rows = evaluate_har_holdout(
            base_path=base_path,
            components_grid=selected_components["HAR"],
            args=args,
        )

    cv_raw_path = output_dir / "lda_components_ablation_cv_raw.csv"
    cv_summary_path = output_dir / "lda_components_ablation_cv_summary.csv"
    holdout_path = output_dir / "lda_components_ablation_holdout.csv"
    config_path = output_dir / "lda_components_ablation_run_config.json"

    cv_raw_df = pd.DataFrame(cv_rows)
    cv_raw_df.to_csv(cv_raw_path, index=False)

    cv_summary_df = summarize_cv(cv_rows)
    cv_summary_df.to_csv(cv_summary_path, index=False)

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(holdout_path, index=False)

    ended_at = now_utc_iso()
    total_elapsed = time.perf_counter() - total_start

    run_config = {
        "created_at": now_utc_iso(),
        "run_started_at": started_at,
        "run_finished_at": ended_at,
        "total_elapsed_seconds": float(total_elapsed),
        "total_elapsed_hms": format_elapsed(total_elapsed),
        "datasets": selected_datasets,
        "components_by_dataset": selected_components,
        "cv_splits_by_dataset": selected_cv_splits,
        "model": {
            "pipeline": "StandardScaler -> LDA(n_components=r) -> GradientBoostingClassifier",
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "subsample": 0.6,
        },
        "seed": args.seed,
        "har_holdout_evaluated": bool("HAR" in selected_datasets and not args.skip_har_holdout),
        "notes": [
            "HAR CV uses official training split only (no test leakage).",
            "HAR holdout uses official train/test files.",
            "YEAST has no official holdout split in this repository; CV only.",
        ],
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[RUN] end: {ended_at} | total elapsed {format_elapsed(total_elapsed)}")
    print("\nAblation complete.")
    print(f"- CV raw: {cv_raw_path}")
    print(f"- CV summary: {cv_summary_path}")
    print(f"- Holdout: {holdout_path}")
    print(f"- Run config: {config_path}")


if __name__ == "__main__":
    main()
