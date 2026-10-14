"""Model factory, timing/size benchmarks and artifact (de)serialisation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_estimator(name: str, seed: int, n_jobs: int, params: dict[str, Any] | None = None):
    """Build an (unfitted) estimator wrapped in a scikit-learn Pipeline where sensible."""
    params = params or {}
    if name == "mean_baseline":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])
    if name == "linear":
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    if name == "ridge":
        return Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=params.get("alpha", 1.0)))]
        )
    if name == "random_forest":
        return Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=params.get("n_estimators", 150),
                        min_samples_leaf=params.get("min_samples_leaf", 5),
                        max_features=params.get("max_features", 0.5),
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                )
            ]
        )
    if name == "hist_gb":
        return Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=params.get("learning_rate", 0.1),
                        max_iter=params.get("max_iter", 300),
                        max_leaf_nodes=params.get("max_leaf_nodes", 63),
                        early_stopping=False,  # sklearn's internal split is a random row split
                        random_state=seed,
                    ),
                )
            ]
        )
    if name == "lightgbm":
        return Pipeline(
            [
                (
                    "model",
                    LGBMRegressor(
                        objective="regression_l2",
                        n_estimators=params.get("n_estimators", 800),
                        learning_rate=params.get("learning_rate", 0.05),
                        num_leaves=params.get("num_leaves", 63),
                        min_child_samples=params.get("min_child_samples", 100),
                        subsample=params.get("subsample", 0.8),
                        subsample_freq=1,
                        colsample_bytree=params.get("colsample_bytree", 0.8),
                        reg_lambda=params.get("reg_lambda", 1.0),
                        n_jobs=n_jobs,
                        random_state=seed,
                        verbose=-1,
                    ),
                )
            ]
        )
    raise ValueError(f"unknown model {name!r}")


@dataclass
class FitRecord:
    """What we know about one fitted candidate."""

    name: str
    feature_set: str
    columns: list[str]
    estimator: Any
    train_rows: int
    fit_seconds: float
    params: dict[str, Any] = field(default_factory=dict)
    best_iteration: int | None = None


def fit_candidate(
    name: str,
    feature_set: str,
    columns: list[str],
    train: pd.DataFrame,
    target: str,
    val: pd.DataFrame | None,
    seed: int,
    n_jobs: int,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 100,
) -> FitRecord:
    """Fit one candidate; LightGBM uses the validation profiles for early stopping."""
    est = make_estimator(name, seed, n_jobs, params)
    x = train[columns]
    y = train[target].to_numpy(dtype="float64")
    best_it = None
    t0 = time.perf_counter()
    if name == "lightgbm" and val is not None:
        from lightgbm import early_stopping, log_evaluation

        model = est.named_steps["model"]
        model.fit(
            x,
            y,
            eval_X=val[columns],
            eval_y=val[target].to_numpy(dtype="float64"),
            eval_metric="l1",
            callbacks=[early_stopping(early_stopping_rounds, verbose=False), log_evaluation(0)],
        )
        best_it = int(model.best_iteration_ or model.n_estimators)
    else:
        est.fit(x, y)
    fit_s = time.perf_counter() - t0
    return FitRecord(
        name, feature_set, list(columns), est, len(train), fit_s, dict(params or {}), best_it
    )


def predict(record_or_est: Any, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    est = record_or_est.estimator if isinstance(record_or_est, FitRecord) else record_or_est
    return np.asarray(est.predict(df[columns]), dtype="float64")


def benchmark_inference(
    est: Any, df: pd.DataFrame, columns: list[str], n: int = 20000
) -> dict[str, float]:
    """Batch throughput and single-row latency (median of repeated calls)."""
    sample = df[columns].iloc[:n]
    t0 = time.perf_counter()
    est.predict(sample)
    batch_s = time.perf_counter() - t0
    row = df[columns].iloc[[0]]
    times = []
    for _ in range(50):
        t1 = time.perf_counter()
        est.predict(row)
        times.append(time.perf_counter() - t1)
    return {
        "batch_rows_per_s": float(len(sample) / batch_s) if batch_s > 0 else float("inf"),
        "single_row_latency_ms": float(np.median(times) * 1e3),
    }


def model_size_kb(est: Any, tmp_path: Path) -> float:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, tmp_path, compress=3)
    size = tmp_path.stat().st_size / 1024.0
    tmp_path.unlink()
    return float(size)


# ---------------------------------------------------------------- artifacts


def save_model_bundle(
    directory: Path,
    stem: str,
    estimator: Any,
    meta: dict[str, Any],
) -> Path:
    """Persist an estimator plus the metadata required to use it safely (columns, history spec)."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{stem}.joblib"
    joblib.dump(estimator, model_path, compress=3)
    (directory / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return model_path


def load_model_bundle(directory: Path, stem: str) -> tuple[Any, dict[str, Any]]:
    model_path = directory / f"{stem}.joblib"
    meta_path = directory / f"{stem}.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"model artifact missing: {model_path} / {meta_path}")
    return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8"))
