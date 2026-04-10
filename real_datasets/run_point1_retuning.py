#!/usr/bin/env python3
"""Point 1 retuning runner for real datasets.

This script prepares and executes an independent retuning protocol for:
- GBM
- PCA+GBM
- LDA+GBM
- LdaBoost

Two scopes are supported:
1) learning-rate-only (recommended first pass, around 0.05)
2) full (broader search over tree hyperparameters + learning rate)

For GBM-based methods, subsample is fixed at 0.6 as requested.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LdaBoost.algorithm import LdaBoost


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    outer_splits: int


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "HAR": DatasetConfig(name="HAR", outer_splits=10),
    "RAINFALL": DatasetConfig(name="RAINFALL", outer_splits=10),
    "IRIS": DatasetConfig(name="IRIS", outer_splits=10),
    "SONAR": DatasetConfig(name="SONAR", outer_splits=10),
    "YEAST": DatasetConfig(name="YEAST", outer_splits=5),
}

ALL_METHODS = ["GBM", "PCA+GBM", "LDA+GBM", "LdaBoost"]
HEARTBEAT_SECONDS = 300
HEARTBEAT_POLL_SECONDS = 5


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_with_heartbeat(label: str, fn):
    """Run a callable in a thread and print elapsed heartbeats every 5 minutes."""
    result = {}
    error = {}
    done = threading.Event()

    start_perf = time.perf_counter()
    start_abs = now_utc_iso()
    print(f"[{label}] start: {start_abs}")

    def _target():
        try:
            result["value"] = fn()
        except Exception as exc:
            error["exc"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    next_tick = start_perf + HEARTBEAT_SECONDS
    while not done.wait(timeout=HEARTBEAT_POLL_SECONDS):
        now_perf = time.perf_counter()
        while now_perf >= next_tick:
            elapsed = now_perf - start_perf
            print(f"[{label}] heartbeat: elapsed {format_elapsed(elapsed)}")
            next_tick += HEARTBEAT_SECONDS

    worker.join()
    end_perf = time.perf_counter()
    end_abs = now_utc_iso()
    print(f"[{label}] end: {end_abs} | elapsed {format_elapsed(end_perf - start_perf)}")

    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def load_har(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Avoid leakage: for nested CV/tuning use only official training split.
    x_train, y_train, _, _ = load_har_train_test(base)
    return x_train, y_train


def load_rainfall_train_test(base: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    train = pd.read_csv(base / "rain" / "Data" / "train.csv")
    test = pd.read_csv(base / "rain" / "Data" / "test.csv")
    y_train = train["rainfall"].astype(int).to_numpy()
    x_train = ensure_numeric_features(train.drop(columns=["rainfall"]))

    # Some Rainfall test files are unlabeled (submission format): keep them usable.
    if "rainfall" in test.columns:
        y_test = test["rainfall"].astype(int).to_numpy()
        x_test = ensure_numeric_features(test.drop(columns=["rainfall"]))
    else:
        y_test = None
        x_test = ensure_numeric_features(test)

    return x_train, y_train, x_test, y_test


def load_rainfall(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Avoid leakage: for nested CV/tuning use only official training split.
    x_train, y_train, _, _ = load_rainfall_train_test(base)
    return x_train, y_train


def load_iris(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(base / "iris" / "Data" / "IRIS.csv")
    y = df["species"].astype(str).to_numpy()
    x = ensure_numeric_features(df.drop(columns=["species"]))
    return x, y


def load_sonar(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(base / "sonar" / "Data" / "sonar.csv", header=None)
    y = df.iloc[:, -1].astype(str).to_numpy()
    x = ensure_numeric_features(df.iloc[:, :-1])
    return x, y


def load_yeast(base: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(base / "yeast" / "Data" / "yeast.csv")
    y = df["name"].astype(str).to_numpy()
    x = ensure_numeric_features(df.drop(columns=["name"]))
    return x, y


def load_dataset(base: Path, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    loaders = {
        "HAR": load_har,
        "RAINFALL": load_rainfall,
        "IRIS": load_iris,
        "SONAR": load_sonar,
        "YEAST": load_yeast,
    }
    return loaders[dataset_name](base)


def safe_n_splits(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    max_feasible = int(np.min(counts))
    return max(2, min(requested, max_feasible))


def make_gbm(random_state: int, n_estimators: int, max_depth: int, min_samples_leaf: int):
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=0.6,
        random_state=random_state,
    )


def tune_with_grid_search(
    estimator,
    param_grid: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
    label: str,
):
    cv_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv_inner,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    run_with_heartbeat(label, lambda: search.fit(x_train, y_train))
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def tune_ldaboost_manual(
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: int,
    random_state: int,
    ldaboost_grid: dict,
    fixed_params: dict,
    label: str,
):
    cv_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = None
    candidates = list(ParameterGrid(ldaboost_grid))
    total_candidates = len(candidates)

    tune_start_perf = time.perf_counter()
    tune_start_abs = now_utc_iso()
    print(f"[{label}] start: {tune_start_abs}")
    next_tick = tune_start_perf + HEARTBEAT_SECONDS

    for idx, candidate in enumerate(candidates, start=1):
        params = {**fixed_params, **candidate}
        fold_scores = []
        for in_train_idx, in_val_idx in cv_inner.split(x_train, y_train):
            x_in_train = x_train[in_train_idx]
            y_in_train = y_train[in_train_idx]
            x_in_val = x_train[in_val_idx]
            y_in_val = y_train[in_val_idx]

            model = LdaBoost(
                n_estimators=int(params["n_estimators"]),
                learning_rate=float(params["learning_rate"]),
                max_depth=int(params["max_depth"]),
                random_state=random_state,
            )
            model.fit(x_in_train, y_in_train)
            y_hat = model.predict(x_in_val)
            fold_scores.append(accuracy_score(y_in_val, y_hat))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        now_perf = time.perf_counter()
        while now_perf >= next_tick:
            elapsed = now_perf - tune_start_perf
            print(
                f"[{label}] heartbeat: elapsed {format_elapsed(elapsed)} "
                f"| candidates {idx}/{total_candidates}"
            )
            next_tick += HEARTBEAT_SECONDS

    assert best_params is not None
    final_model = LdaBoost(
        n_estimators=int(best_params["n_estimators"]),
        learning_rate=float(best_params["learning_rate"]),
        max_depth=int(best_params["max_depth"]),
        random_state=random_state,
    )
    run_with_heartbeat(f"{label} refit", lambda: final_model.fit(x_train, y_train))

    tune_end_perf = time.perf_counter()
    tune_end_abs = now_utc_iso()
    print(f"[{label}] end: {tune_end_abs} | elapsed {format_elapsed(tune_end_perf - tune_start_perf)}")
    return final_model, best_params, best_score


def build_tuning_spaces(args, learning_rates, full_n_estimators, full_max_depth, full_min_samples_leaf):
    if args.scope == "learning-rate-only":
        gb_fixed = {
            "n_estimators": args.base_n_estimators,
            "max_depth": args.base_max_depth,
            "min_samples_leaf": args.base_min_samples_leaf,
        }
        gb_grid = {"learning_rate": learning_rates}

        ldaboost_fixed = {
            "n_estimators": args.base_n_estimators,
            "max_depth": args.base_max_depth,
        }
        ldaboost_grid = {"learning_rate": learning_rates}
    else:
        gb_fixed = {
            "n_estimators": args.base_n_estimators,
            "max_depth": args.base_max_depth,
            "min_samples_leaf": args.base_min_samples_leaf,
        }
        gb_grid = {
            "n_estimators": full_n_estimators,
            "max_depth": full_max_depth,
            "min_samples_leaf": full_min_samples_leaf,
            "learning_rate": learning_rates,
        }

        ldaboost_fixed = {
            "n_estimators": args.base_n_estimators,
            "max_depth": args.base_max_depth,
        }
        ldaboost_grid = {
            "n_estimators": full_n_estimators,
            "max_depth": full_max_depth,
            "learning_rate": learning_rates,
        }

    return gb_fixed, gb_grid, ldaboost_fixed, ldaboost_grid


def tune_method(
    method_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: int,
    args,
    gb_fixed: dict,
    gb_grid: dict,
    ldaboost_fixed: dict,
    ldaboost_grid: dict,
    label: str,
):
    if method_name == "GBM":
        base = make_gbm(args.seed, **gb_fixed)
        return tune_with_grid_search(
            estimator=base,
            param_grid=gb_grid,
            x_train=x_train,
            y_train=y_train,
            inner_splits=inner_splits,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            label=label,
        )

    if method_name == "PCA+GBM":
        base = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("gb", make_gbm(args.seed, **gb_fixed)),
            ]
        )
        pca_grid = {f"gb__{k}": v for k, v in gb_grid.items()}
        return tune_with_grid_search(
            estimator=base,
            param_grid=pca_grid,
            x_train=x_train,
            y_train=y_train,
            inner_splits=inner_splits,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            label=label,
        )

    if method_name == "LDA+GBM":
        base = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis(n_components=None)),
                ("gb", make_gbm(args.seed, **gb_fixed)),
            ]
        )
        lda_grid = {f"gb__{k}": v for k, v in gb_grid.items()}
        return tune_with_grid_search(
            estimator=base,
            param_grid=lda_grid,
            x_train=x_train,
            y_train=y_train,
            inner_splits=inner_splits,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            label=label,
        )

    return tune_ldaboost_manual(
        x_train=x_train,
        y_train=y_train,
        inner_splits=inner_splits,
        random_state=args.seed,
        ldaboost_grid=ldaboost_grid,
        fixed_params=ldaboost_fixed,
        label=label,
    )


def compute_summary_rows(fold_accuracies: Dict[str, Dict[str, List[float]]]) -> List[dict]:
    rows: List[dict] = []
    for dataset_name, method_map in fold_accuracies.items():
        for method_name, accs in method_map.items():
            if not accs:
                continue
            arr = np.array(accs, dtype=float)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "mean_accuracy": float(np.mean(arr)),
                    "std_accuracy": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n_folds": int(len(arr)),
                }
            )
    return rows


def persist_outputs(
    output_dir: Path,
    fold_accuracies: Dict[str, Dict[str, List[float]]],
    best_params_rows: List[dict],
    fold_timing_rows: List[dict],
    holdout_test_rows: List[dict],
    total_start_abs: str,
    selected_datasets: List[str],
    selected_methods: List[str],
    args,
    learning_rates: List[float],
    completed_datasets: List[str],
    run_stage: str,
    run_finished_abs: Optional[str] = None,
    total_elapsed: Optional[float] = None,
) -> None:
    fold_json_path = output_dir / "point1_retune_fold_accuracies.json"
    summary_csv_path = output_dir / "point1_retune_fold_accuracy_summary.csv"
    best_csv_path = output_dir / "point1_retune_best_params.csv"
    fold_timing_csv_path = output_dir / "point1_retune_fold_timing.csv"
    holdout_csv_path = output_dir / "point1_retune_holdout_test_accuracy.csv"
    config_json_path = output_dir / "point1_retune_run_config.json"

    with fold_json_path.open("w", encoding="utf-8") as f:
        json.dump(fold_accuracies, f, indent=2)

    summary_rows = compute_summary_rows(fold_accuracies)
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    pd.DataFrame(best_params_rows).to_csv(best_csv_path, index=False)
    pd.DataFrame(fold_timing_rows).to_csv(fold_timing_csv_path, index=False)
    pd.DataFrame(holdout_test_rows).to_csv(holdout_csv_path, index=False)

    run_cfg = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_started_at": total_start_abs,
        "run_finished_at": run_finished_abs,
        "total_elapsed_seconds": float(total_elapsed) if total_elapsed is not None else None,
        "total_elapsed_hms": format_elapsed(total_elapsed) if total_elapsed is not None else None,
        "run_stage": run_stage,
        "completed_datasets": completed_datasets,
        "cv_data_policy": {
            "HAR": "train_only_for_tuning_and_cv",
            "RAINFALL": "train_only_for_tuning_and_cv",
            "IRIS": "full_dataset_cv",
            "SONAR": "full_dataset_cv",
            "YEAST": "full_dataset_cv",
        },
        "datasets": selected_datasets,
        "methods": selected_methods,
        "scope": args.scope,
        "learning_rates": learning_rates,
        "inner_splits": args.inner_splits,
        "outer_splits_override": args.outer_splits,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "gbm_subsample": 0.6,
        "holdout_test_datasets": ["HAR", "RAINFALL"],
        "note": "LdaBoost class does not expose subsample; only GBM-based pipelines fix subsample=0.6.",
    }
    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Point 1 retuning on real datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="HAR,RAINFALL,IRIS,SONAR,YEAST",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--scope",
        choices=["learning-rate-only", "full"],
        default="learning-rate-only",
        help="Retuning scope.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="0.03,0.05,0.07,0.10",
        help="Comma-separated learning rates to test.",
    )
    parser.add_argument("--inner-splits", type=int, default=5, help="Inner CV folds.")
    parser.add_argument(
        "--outer-splits",
        type=int,
        default=None,
        help="Override outer CV folds for all datasets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "real_datasets" / "point1_outputs"),
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--base-n-estimators",
        type=int,
        default=100,
        help="Default n_estimators for learning-rate-only scope.",
    )
    parser.add_argument(
        "--base-max-depth",
        type=int,
        default=3,
        help="Default max_depth for learning-rate-only scope.",
    )
    parser.add_argument(
        "--base-min-samples-leaf",
        type=int,
        default=1,
        help="Default min_samples_leaf for learning-rate-only scope (GBM only).",
    )
    parser.add_argument(
        "--full-n-estimators",
        type=str,
        default="100,200,300",
        help="Grid for n_estimators in full scope.",
    )
    parser.add_argument(
        "--full-max-depth",
        type=str,
        default="2,3,4",
        help="Grid for max_depth in full scope.",
    )
    parser.add_argument(
        "--full-min-samples-leaf",
        type=str,
        default="1,5,10",
        help="Grid for min_samples_leaf in full scope (GBM only).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help="Comma-separated methods to run. Allowed: GBM,PCA+GBM,LDA+GBM,LdaBoost",
    )
    args = parser.parse_args()

    selected_datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in selected_datasets if d not in DATASET_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed: {list(DATASET_CONFIGS.keys())}")

    selected_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown_methods = [m for m in selected_methods if m not in ALL_METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Allowed: {ALL_METHODS}")

    learning_rates = parse_float_list(args.learning_rates)
    full_n_estimators = parse_int_list(args.full_n_estimators)
    full_max_depth = parse_int_list(args.full_max_depth)
    full_min_samples_leaf = parse_int_list(args.full_min_samples_leaf)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start_perf = time.perf_counter()
    total_start_abs = now_utc_iso()
    print(f"[RUN] start: {total_start_abs}")

    gb_fixed, gb_grid, ldaboost_fixed, ldaboost_grid = build_tuning_spaces(
        args=args,
        learning_rates=learning_rates,
        full_n_estimators=full_n_estimators,
        full_max_depth=full_max_depth,
        full_min_samples_leaf=full_min_samples_leaf,
    )

    fold_accuracies: Dict[str, Dict[str, List[float]]] = {}
    best_params_rows = []
    fold_timing_rows = []
    holdout_test_rows = []
    completed_datasets: List[str] = []

    for dataset_name in selected_datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        x_raw, y_raw = load_dataset(PROJECT_ROOT / "real_datasets", dataset_name)
        y_enc = LabelEncoder().fit_transform(y_raw)

        cfg = DATASET_CONFIGS[dataset_name]
        requested_outer = args.outer_splits if args.outer_splits is not None else cfg.outer_splits
        outer_splits = safe_n_splits(y_enc, requested_outer)

        cv_outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=args.seed)
        fold_accuracies[dataset_name] = {method_name: [] for method_name in selected_methods}

        for fold_id, (train_idx, test_idx) in enumerate(cv_outer.split(x_raw, y_enc), start=1):
            fold_label = f"{dataset_name} fold {fold_id}/{outer_splits}"
            fold_start_perf = time.perf_counter()
            fold_start_abs = now_utc_iso()
            print(f"[{fold_label}] start: {fold_start_abs}")
            x_train = x_raw[train_idx]
            y_train = y_enc[train_idx]
            x_test = x_raw[test_idx]
            y_test = y_enc[test_idx]

            inner_splits = safe_n_splits(y_train, args.inner_splits)

            for method_name in selected_methods:
                t0 = time.perf_counter()

                best_model, best_params, best_inner_score = tune_method(
                    method_name=method_name,
                    x_train=x_train,
                    y_train=y_train,
                    inner_splits=inner_splits,
                    args=args,
                    gb_fixed=gb_fixed,
                    gb_grid=gb_grid,
                    ldaboost_fixed=ldaboost_fixed,
                    ldaboost_grid=ldaboost_grid,
                    label=f"{fold_label} | {method_name}",
                )

                y_hat = best_model.predict(x_test)
                fold_acc = float(accuracy_score(y_test, y_hat))
                elapsed = time.perf_counter() - t0

                fold_accuracies[dataset_name][method_name].append(fold_acc)
                best_params_rows.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold_id,
                        "method": method_name,
                        "inner_splits": inner_splits,
                        "outer_accuracy": fold_acc,
                        "best_inner_cv_accuracy": float(best_inner_score),
                        "runtime_seconds": elapsed,
                        "best_params": json.dumps(best_params, sort_keys=True),
                    }
                )

            fold_end_perf = time.perf_counter()
            fold_end_abs = now_utc_iso()
            fold_elapsed = fold_end_perf - fold_start_perf
            print(f"[{fold_label}] end: {fold_end_abs} | elapsed {format_elapsed(fold_elapsed)}")
            fold_timing_rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_id,
                    "fold_label": fold_label,
                    "start_utc": fold_start_abs,
                    "end_utc": fold_end_abs,
                    "elapsed_seconds": float(fold_elapsed),
                    "elapsed_hms": format_elapsed(fold_elapsed),
                }
            )

        completed_datasets.append(dataset_name)
        persist_outputs(
            output_dir=output_dir,
            fold_accuracies=fold_accuracies,
            best_params_rows=best_params_rows,
            fold_timing_rows=fold_timing_rows,
            holdout_test_rows=holdout_test_rows,
            total_start_abs=total_start_abs,
            selected_datasets=selected_datasets,
            selected_methods=selected_methods,
            args=args,
            learning_rates=learning_rates,
            completed_datasets=completed_datasets,
            run_stage=f"dataset_completed:{dataset_name}",
        )
        print(f"[CHECKPOINT] wrote partial results after dataset {dataset_name}")

    print("\n=== Holdout test evaluation (HAR, RAINFALL) ===")
    holdout_start_perf = time.perf_counter()
    holdout_start_abs = now_utc_iso()
    print(f"[HOLDOUT] start: {holdout_start_abs}")
    holdout_loaders = {
        "HAR": load_har_train_test,
        "RAINFALL": load_rainfall_train_test,
    }
    for dataset_name, loader in holdout_loaders.items():
        x_train, y_train_raw, x_test, y_test_raw = loader(PROJECT_ROOT / "real_datasets")

        if y_test_raw is None:
            print(
                f"[HOLDOUT] skip {dataset_name}: test target column is missing; "
                f"holdout test accuracy cannot be computed."
            )
            continue

        encoder = LabelEncoder()
        encoder.fit(np.concatenate([y_train_raw, y_test_raw], axis=0))
        y_train = encoder.transform(y_train_raw)
        y_test = encoder.transform(y_test_raw)
        inner_splits = safe_n_splits(y_train, args.inner_splits)

        for method_name in selected_methods:
            eval_label = f"{dataset_name} holdout | {method_name}"
            t0 = time.perf_counter()
            method_start_abs = now_utc_iso()
            print(f"[{eval_label}] validation accuracy computation start: {method_start_abs}")

            best_model, best_params, best_inner_score = tune_method(
                method_name=method_name,
                x_train=x_train,
                y_train=y_train,
                inner_splits=inner_splits,
                args=args,
                gb_fixed=gb_fixed,
                gb_grid=gb_grid,
                ldaboost_fixed=ldaboost_fixed,
                ldaboost_grid=ldaboost_grid,
                label=eval_label,
            )

            y_hat = best_model.predict(x_test)
            test_acc = float(accuracy_score(y_test, y_hat))
            elapsed = time.perf_counter() - t0
            print(
                f"[{eval_label}] test accuracy: {test_acc:.6f} "
                f"| elapsed {format_elapsed(elapsed)}"
            )

            holdout_test_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "train_size": int(len(y_train)),
                    "test_size": int(len(y_test)),
                    "inner_splits": inner_splits,
                    "test_accuracy": test_acc,
                    "best_inner_cv_accuracy": float(best_inner_score),
                    "runtime_seconds": elapsed,
                    "best_params": json.dumps(best_params, sort_keys=True),
                }
            )

    holdout_end_perf = time.perf_counter()
    holdout_end_abs = now_utc_iso()
    print(
        f"[HOLDOUT] end: {holdout_end_abs} "
        f"| elapsed {format_elapsed(holdout_end_perf - holdout_start_perf)}"
    )

    total_end_perf = time.perf_counter()
    total_end_abs = now_utc_iso()
    total_elapsed = total_end_perf - total_start_perf

    print(f"[RUN] end: {total_end_abs} | total elapsed {format_elapsed(total_elapsed)}")

    persist_outputs(
        output_dir=output_dir,
        fold_accuracies=fold_accuracies,
        best_params_rows=best_params_rows,
        fold_timing_rows=fold_timing_rows,
        holdout_test_rows=holdout_test_rows,
        total_start_abs=total_start_abs,
        selected_datasets=selected_datasets,
        selected_methods=selected_methods,
        args=args,
        learning_rates=learning_rates,
        completed_datasets=completed_datasets,
        run_stage="finished",
        run_finished_abs=total_end_abs,
        total_elapsed=total_elapsed,
    )

    fold_json_path = output_dir / "point1_retune_fold_accuracies.json"
    summary_csv_path = output_dir / "point1_retune_fold_accuracy_summary.csv"
    best_csv_path = output_dir / "point1_retune_best_params.csv"
    fold_timing_csv_path = output_dir / "point1_retune_fold_timing.csv"
    holdout_csv_path = output_dir / "point1_retune_holdout_test_accuracy.csv"
    config_json_path = output_dir / "point1_retune_run_config.json"

    print("\nRun complete.")
    print(f"- Fold accuracies: {fold_json_path}")
    print(f"- Summary CSV: {summary_csv_path}")
    print(f"- Best params CSV: {best_csv_path}")
    print(f"- Fold timing CSV: {fold_timing_csv_path}")
    print(f"- Holdout test accuracy CSV: {holdout_csv_path}")
    print(f"- Run config: {config_json_path}")


if __name__ == "__main__":
    main()