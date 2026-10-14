"""Loading trained artifacts and producing estimates (batch, single-sample, streaming).

Used by the Streamlit app and tests. Models only ever see the columns listed in their metadata;
extra columns (including measured temperatures) are ignored by construction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import features as feat
from . import models as mdl
from .config import ARTIFACTS_DIR, MODES, mode_dir

RAW_INPUTS = feat.RAW_FEATURES


def instant_features_scalar(sample: Mapping[str, float]) -> dict[str, float]:
    """Pure-Python twin of :func:`features.add_instant_features` for one sample."""
    i_d, i_q = float(sample["i_d"]), float(sample["i_q"])
    u_d, u_q = float(sample["u_d"]), float(sample["u_q"])
    speed, torque = float(sample["motor_speed"]), float(sample["torque"])
    i_sq = i_d**2 + i_q**2
    p_mech = torque * speed * (2.0 * math.pi / 60.0)
    p_elec = 1.5 * (u_d * i_d + u_q * i_q)
    out = {k: float(sample[k]) for k in RAW_INPUTS}
    out.update(
        i_sq=i_sq,
        i_mag=math.sqrt(i_sq),
        u_mag=math.sqrt(u_d**2 + u_q**2),
        abs_speed=abs(speed),
        abs_torque=abs(torque),
        p_mech_w=p_mech,
        p_elec_proxy=p_elec,
        p_loss_proxy=p_elec - p_mech,
        ambient_minus_coolant=float(sample["ambient"]) - float(sample["coolant"]),
    )
    return out


def available_modes() -> list[str]:
    """Modes that have both trained model bundles on disk (``full`` is listed first)."""
    found = []
    for mode in ("full", "fast"):
        d = mode_dir(mode) / "models"
        if (d / "track_a.joblib").is_file() and (d / "track_b.joblib").is_file():
            found.append(mode)
    return found


class VirtualSensor:
    """A trained winding-temperature estimator with its feature recipe."""

    def __init__(self, estimator: Any, meta: dict[str, Any]):
        self.estimator = estimator
        self.meta = meta
        self.columns: list[str] = list(meta["columns"])
        self.spec: dict[str, Any] | None = meta.get("history_spec")
        self.track: str = meta["track"]

    @classmethod
    def load(cls, mode: str, track: str) -> VirtualSensor:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        est, meta = mdl.load_model_bundle(mode_dir(mode) / "models", f"track_{track.lower()}")
        return cls(est, meta)

    @property
    def uses_history(self) -> bool:
        return bool(self.spec and self.spec["signals"])

    def frame_features(self, raw: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
        """Feature frame for one or many profiles. Without ``group`` the rows are ONE profile."""
        missing = [c for c in RAW_INPUTS if c not in raw.columns]
        if missing:
            raise KeyError(f"missing input columns: {missing}")
        df = feat.add_instant_features(raw)
        if self.uses_history:
            key = group or "__single__"
            if key not in df.columns:
                df[key] = 0
            df = feat.add_history_features(df, key, **self.spec)
        return df

    def predict_frame(self, raw: pd.DataFrame, group: str | None = None) -> np.ndarray:
        df = self.frame_features(raw, group)
        return np.asarray(self.estimator.predict(df[self.columns]), dtype="float64")

    def predict_one(self, sample: Mapping[str, float]) -> float:
        """Instantaneous estimate (Track A only - needs no history)."""
        if self.uses_history:
            raise ValueError("predict_one is only valid for the instantaneous Track A model")
        row = pd.DataFrame([instant_features_scalar(sample)])
        return float(self.estimator.predict(row[self.columns])[0])

    def stream(self) -> StreamingPredictor:
        return StreamingPredictor(self)


class StreamingPredictor:
    """Sample-by-sample estimates with an internal causal history buffer (live-style usage)."""

    def __init__(self, sensor: VirtualSensor):
        self.sensor = sensor
        self.hist = (
            feat.StreamingHistory(**sensor.spec) if sensor.uses_history else None  # type: ignore[arg-type]
        )

    def reset(self) -> None:
        if self.hist:
            self.hist.reset()

    def update(self, sample: Mapping[str, float]) -> float:
        row = instant_features_scalar(sample)
        if self.hist:
            row.update(self.hist.update(row))
        x = pd.DataFrame([row])[self.sensor.columns]
        return float(self.sensor.estimator.predict(x)[0])


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_results(mode: str) -> dict[str, Any] | None:
    return load_json(mode_dir(mode) / "metrics_test.json")


def load_input_ranges(mode: str) -> dict[str, dict[str, float]] | None:
    return load_json(mode_dir(mode) / "input_ranges.json")


def check_input_ranges(
    sample: Mapping[str, float], ranges: Mapping[str, Mapping[str, float]] | None
) -> list[str]:
    """Human-readable warnings for inputs outside the span of the training data."""
    if not ranges:
        return []
    out = []
    for k, r in ranges.items():
        if k in sample and not (r["min"] <= float(sample[k]) <= r["max"]):
            out.append(
                f"{k}={float(sample[k]):.2f} outside training range [{r['min']:.1f}, {r['max']:.1f}]"
            )
    return out


__all__ = ["ARTIFACTS_DIR", "StreamingPredictor", "VirtualSensor", "available_modes"]
