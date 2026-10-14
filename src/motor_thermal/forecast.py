"""Optional extension: forecast the winding temperature ``horizon_s`` seconds AHEAD.

Kept separate from the contemporaneous estimator. Target = winding temperature at t + h, shifted
within each profile (rows near the end of a profile have no target and are dropped). Features use
information available at time t only - no future current, speed, torque, coolant or temperature.

Two variants are compared against a persistence baseline (T(t+h) = T(t)):

* F1 "sensor-independent": the same features as the Track B estimator (no winding measurement).
* F2 "sensor-assisted": F1 plus the *measured* winding temperature at t and its change over the
  last h seconds (causal). It needs the physical sensor - a different use case.

The contemporaneous Track B estimate is also scored against the future truth, to show that a
current-temperature estimator is not a forecaster.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from . import metrics as met
from . import models as mdl
from . import plots
from . import splits as spl
from .config import FIGURES_DIR, mode_dir
from .evaluate import dump_json
from .train import load_everything, thin_training_rows

log = logging.getLogger(__name__)

_TUNABLE = (
    "num_leaves",
    "learning_rate",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
)


def add_future_target(frame: pd.DataFrame, group: str, target: str, h: int) -> pd.DataFrame:
    """Add ``y_future`` (value h samples ahead, same profile), ``y_now_lag`` and drop rows without it."""
    out = frame.copy()
    g = out.groupby(group, sort=False)[target]
    out["y_future"] = g.shift(-h)
    out["w_now"] = out[target]
    out["w_change_h"] = out[target] - g.shift(h)  # causal: uses the PAST value
    return out.dropna(subset=["y_future"])


def run_forecast(cfg: dict[str, Any], mode: str, csv: str | Path | None = None) -> dict[str, Any]:
    out_dir = mode_dir(mode)
    group, target = cfg["columns"]["group"], cfg["columns"]["target"]
    hz = float(cfg["dataset"]["sampling_hz"])
    h = int(cfg["forecast"]["horizon_s"] * hz)
    ctx = load_everything(cfg, csv)
    frame, manifest = ctx["frame"], ctx["manifest"]
    frame = add_future_target(frame, group, target, h)

    est_b, meta_b = mdl.load_model_bundle(out_dir / "models", "track_b")
    cols_f1 = list(meta_b["columns"])
    cols_f2 = [*cols_f1, "w_now", "w_change_h"]
    train, _ = thin_training_rows(frame, manifest, cfg, mode)
    val = spl.split_frame(frame, manifest, "val", group)
    test = spl.split_frame(frame, manifest, "test", group)
    params = {k: v for k, v in meta_b.get("params", {}).items() if k != "n_estimators"}

    results: dict[str, Any] = {
        "horizon_s": cfg["forecast"]["horizon_s"],
        "target": "stator_winding at t + horizon (shifted within profile)",
        "rows_test": int(len(test)),
        "note": "Evaluated on the same held-out test profiles; rows in the last horizon seconds of "
        "each profile are excluded because they have no future label.",
    }
    y_test = test["y_future"].to_numpy("float64")
    pid_test = test[group].to_numpy()

    def score(name: str, pred: np.ndarray) -> dict[str, Any]:
        s = met.summarize_all(pid_test, y_test, pred)
        results[name] = s
        log.info(
            "[forecast %s] test MAE %.3f RMSE %.3f",
            name,
            s["aggregate"]["mae"],
            s["aggregate"]["rmse"],
        )
        return s

    score("persistence_T_now", test["w_now"].to_numpy("float64"))
    score(
        "track_B_estimate_used_as_forecast",
        np.asarray(est_b.predict(test[cols_f1]), dtype="float64"),
    )
    for name, cols in (("F1_sensor_independent", cols_f1), ("F2_sensor_assisted", cols_f2)):
        hp = {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.0,
            **{k: v for k, v in params.items() if k in _TUNABLE},
        }
        m = LGBMRegressor(
            n_estimators=cfg["training"][mode]["lgbm_max_rounds"],
            subsample_freq=1,
            n_jobs=cfg["training"]["n_jobs"],
            random_state=cfg["seed"],
            verbose=-1,
            **hp,
        )
        m.fit(
            train[cols],
            train["y_future"].to_numpy("float64"),
            eval_X=val[cols],
            eval_y=val["y_future"].to_numpy("float64"),
            eval_metric="l1",
            callbacks=[early_stopping(cfg["training"]["early_stopping_rounds"], verbose=False)],
        )
        score(name, np.asarray(m.predict(test[cols]), dtype="float64"))
        results[name]["best_iteration"] = int(m.best_iteration_)

    labels = {
        "persistence_T_now": "persistence\nT(t)",
        "track_B_estimate_used_as_forecast": "Track B\nestimate\n(not a forecaster)",
        "F1_sensor_independent": "F1 forecast\nno sensor",
        "F2_sensor_assisted": "F2 forecast\n+ sensor T(t)",
    }
    mae = {labels[k]: results[k]["aggregate"]["mae"] for k in labels}
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        list(mae), list(mae.values()), color=[plots.C_GREY, plots.C_B, plots.C_A, plots.C_OK]
    )
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
        )
    ax.set(
        ylabel="MAE vs. T(t + 60 s) on test [°C]",
        title="60 s ahead forecasting (separate experiment)",
    )
    plots._save(fig, FIGURES_DIR / mode / "forecast_comparison.png")
    dump_json(results, out_dir / "forecast_test.json")
    return results


__all__ = ["run_forecast"]
