"""Regression metrics, profile-level summaries and stratified error tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R2 plus robust and tail error statistics (all in deg C except R2)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "median_ae": float(np.median(abs_err)),
        "p95_ae": float(np.percentile(abs_err, 95)),
        "max_ae": float(abs_err.max()),
        "bias": float(err.mean()),
        "n": int(y_true.size),
    }


def per_profile_metrics(
    profile_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """One metrics row per profile (profile-level view; every profile counts equally)."""
    rows = []
    for pid in pd.unique(profile_ids):
        m = profile_ids == pid
        rows.append({"profile_id": int(pid), **regression_metrics(y_true[m], y_pred[m])})
    return pd.DataFrame(rows).set_index("profile_id")


def summarize_profiles(per_profile: pd.DataFrame) -> dict[str, float]:
    """Profile-averaged MAE (macro) and worst-profile behaviour."""
    worst = per_profile["mae"].idxmax()
    return {
        "macro_mae": float(per_profile["mae"].mean()),
        "median_profile_mae": float(per_profile["mae"].median()),
        "worst_profile_mae": float(per_profile["mae"].max()),
        "worst_profile_id": int(worst),
        "best_profile_mae": float(per_profile["mae"].min()),
        "n_profiles": int(len(per_profile)),
    }


def error_by_bins(
    values: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, edges: list[float], label: str
) -> pd.DataFrame:
    """MAE/RMSE/p95 within value bins (e.g. actual temperature ranges), ordered by bin edge."""
    cut = pd.cut(pd.Series(values), bins=edges, include_lowest=True)
    err = np.asarray(y_pred, dtype="float64") - np.asarray(y_true, dtype="float64")
    df = pd.DataFrame({label: cut, "abs_err": np.abs(err), "err": err, "sq": err**2})
    grp = df.groupby(label, observed=True)
    out = grp.agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        p95_ae=("abs_err", lambda s: float(np.percentile(s, 95))),
        bias=("err", "mean"),
        mse=("sq", "mean"),
    )
    out["rmse"] = np.sqrt(out.pop("mse"))
    out = out.reset_index()
    out[label] = out[label].astype(str)
    return out


def operating_region_table(
    abs_speed: np.ndarray,
    abs_torque: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speed_edges: list[float],
    torque_edges: list[float],
) -> pd.DataFrame:
    """MAE in a speed x torque grid (edges are derived from TRAINING quantiles by the caller)."""
    s = pd.cut(
        pd.Series(abs_speed), bins=speed_edges, include_lowest=True, labels=["low", "mid", "high"]
    )
    t = pd.cut(
        pd.Series(abs_torque), bins=torque_edges, include_lowest=True, labels=["low", "mid", "high"]
    )
    err = np.abs(np.asarray(y_pred, dtype="float64") - np.asarray(y_true, dtype="float64"))
    df = pd.DataFrame(
        {"speed_region": s.astype(str), "torque_region": t.astype(str), "abs_err": err}
    )
    out = df.groupby(["speed_region", "torque_region"], observed=True).agg(
        n=("abs_err", "size"), mae=("abs_err", "mean")
    )
    return out.reset_index()


def quantile_edges(values: np.ndarray) -> list[float]:
    """Tertile edges (min, q33, q67, max) used for low/mid/high operating regions."""
    q = np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])
    q[0], q[-1] = -np.inf, np.inf
    return [float(x) for x in q]


def summarize_all(
    profile_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Aggregate (sample-weighted) plus per-profile summary in one dict."""
    pp = per_profile_metrics(profile_ids, y_true, y_pred)
    return {
        "aggregate": regression_metrics(y_true, y_pred),
        "profile_level": summarize_profiles(pp),
    }
