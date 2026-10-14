"""Feature engineering: instantaneous physics features and strictly causal history features.

Two implementations of the history features are provided and unit-tested to agree:

* :func:`add_history_features` - vectorised, per-profile, used for training/evaluation.
* :class:`StreamingHistory` - sample-by-sample, used for live-style inference.

Causality: the value at sample ``t`` only depends on samples ``<= t`` of the *same* profile.
No winding/tooth/yoke/magnet temperature (or lag of it) is ever used.

Units assumed for the Paderborn ``measures_v2`` file (plausible from value ranges; the
files ship no data dictionary): speed [rpm], torque [Nm], currents [A], voltages [V],
temperatures [deg C].
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- instantaneous features

#: exact physical quantities (given the unit assumptions) vs proxies - documented for the report.
FEATURE_KIND = {
    "i_sq": "exact: |i|^2 = i_d^2 + i_q^2 (copper-loss driver; amplitude convention unknown)",
    "i_mag": "exact: sqrt(i_d^2 + i_q^2)",
    "u_mag": "exact: sqrt(u_d^2 + u_q^2)",
    "abs_speed": "exact: |motor_speed|",
    "abs_torque": "exact: |torque|",
    "p_mech_w": "exact if torque [Nm] and speed [rpm]: torque * speed * 2*pi/60",
    "p_elec_proxy": "proxy: 1.5*(u_d*i_d + u_q*i_q); factor assumes amplitude-invariant dq",
    "p_loss_proxy": "proxy: p_elec_proxy - p_mech_w (total loss estimate, ~copper+iron+friction)",
    "ambient_minus_coolant": "exact: ambient - coolant [K]",
}

RAW_FEATURES = ["ambient", "coolant", "motor_speed", "torque", "i_d", "i_q", "u_d", "u_q"]
INSTANT_ENHANCED = list(FEATURE_KIND)
BASIC_SET = list(RAW_FEATURES)
ENHANCED_SET = RAW_FEATURES + INSTANT_ENHANCED


def add_instant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with row-wise derived features (no history involved)."""
    out = df.copy()
    i_d = out["i_d"].astype("float64")
    i_q = out["i_q"].astype("float64")
    u_d = out["u_d"].astype("float64")
    u_q = out["u_q"].astype("float64")
    speed = out["motor_speed"].astype("float64")
    torque = out["torque"].astype("float64")
    i_sq = i_d**2 + i_q**2
    p_mech = torque * speed * (2.0 * math.pi / 60.0)
    p_elec = 1.5 * (u_d * i_d + u_q * i_q)
    derived = {
        "i_sq": i_sq,
        "i_mag": np.sqrt(i_sq),
        "u_mag": np.sqrt(u_d**2 + u_q**2),
        "abs_speed": speed.abs(),
        "abs_torque": torque.abs(),
        "p_mech_w": p_mech,
        "p_elec_proxy": p_elec,
        "p_loss_proxy": p_elec - p_mech,
        "ambient_minus_coolant": out["ambient"].astype("float64")
        - out["coolant"].astype("float64"),
    }
    for name, values in derived.items():
        out[name] = values.astype("float32")
    return out


# ---------------------------------------------------------------- history features


def _tau_to_alpha(tau_s: float, hz: float) -> float:
    """Per-sample EWMA weight for a time constant ``tau_s`` seconds: alpha = 1 - exp(-dt/tau)."""
    return 1.0 - math.exp(-1.0 / (tau_s * hz))


def history_columns(
    signals: Iterable[str], ewma_tau_s: Iterable[float], rolling_s: Iterable[float]
) -> list[str]:
    cols: list[str] = []
    for s in signals:
        cols += [f"ewm_{s}_{int(t)}s" for t in ewma_tau_s]
        cols += [f"roll_{s}_{int(w)}s" for w in rolling_s]
    return cols


def add_history_features(
    df: pd.DataFrame,
    group: str,
    signals: list[str],
    ewma_tau_s: list[float],
    rolling_s: list[float],
    hz: float,
) -> pd.DataFrame:
    """Causal EWMA and rolling-mean features, computed independently per profile.

    * EWMA uses ``adjust=True`` (a properly normalised weighted mean of the samples seen so far),
      so it is defined from the first sample with no arbitrary initial value.
    * Rolling means require a full window (``min_periods == window``); during the first
      ``window - 1`` samples they are NaN - the cold start is explicit, not filled.
    * Rows must be in chronological order within each profile (validated in ``data``).
    """
    out = df.copy()
    new_cols = history_columns(signals, ewma_tau_s, rolling_s)
    if not new_cols:
        return out
    blocks = {c: np.full(len(out), np.nan, dtype="float32") for c in new_cols}
    group_values = out[group].to_numpy()
    # profiles are contiguous blocks; find block boundaries
    bounds = np.flatnonzero(np.r_[True, group_values[1:] != group_values[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        for s in signals:
            x = pd.Series(out[s].to_numpy(dtype="float64")[a:b])
            for tau in ewma_tau_s:
                alpha = _tau_to_alpha(tau, hz)
                blocks[f"ewm_{s}_{int(tau)}s"][a:b] = (
                    x.ewm(alpha=alpha, adjust=True).mean().to_numpy(dtype="float32")
                )
            for w in rolling_s:
                n = max(1, int(round(w * hz)))
                blocks[f"roll_{s}_{int(w)}s"][a:b] = (
                    x.rolling(n, min_periods=n).mean().to_numpy(dtype="float32")
                )
    for c, v in blocks.items():
        out[c] = v
    return out


class StreamingHistory:
    """Sample-by-sample twin of :func:`add_history_features` for one profile / one live stream.

    ``update(sample)`` takes the *instantaneous-feature* values (it needs the ``signals``) and
    returns the history feature values for that time step. ``reset()`` starts a new profile.
    """

    def __init__(
        self, signals: list[str], ewma_tau_s: list[float], rolling_s: list[float], hz: float
    ):
        self.signals = list(signals)
        self.ewma_tau_s = list(ewma_tau_s)
        self.rolling_s = list(rolling_s)
        self.hz = hz
        self.columns = history_columns(signals, ewma_tau_s, rolling_s)
        self._decay = {t: 1.0 - _tau_to_alpha(t, hz) for t in self.ewma_tau_s}
        self._win = {w: max(1, int(round(w * hz))) for w in self.rolling_s}
        self.reset()

    def reset(self) -> None:
        self._num = {(s, t): 0.0 for s in self.signals for t in self.ewma_tau_s}
        self._den = {(s, t): 0.0 for s in self.signals for t in self.ewma_tau_s}
        maxw = max(self._win.values(), default=1)
        self._buf = {s: deque(maxlen=maxw) for s in self.signals}
        self.n_seen = 0

    def update(self, sample: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        self.n_seen += 1
        for s in self.signals:
            x = float(sample[s])
            self._buf[s].append(x)
            for t in self.ewma_tau_s:
                d = self._decay[t]
                self._num[(s, t)] = d * self._num[(s, t)] + x
                self._den[(s, t)] = d * self._den[(s, t)] + 1.0
                out[f"ewm_{s}_{int(t)}s"] = self._num[(s, t)] / self._den[(s, t)]
            for w, n in self._win.items():
                buf = self._buf[s]
                out[f"roll_{s}_{int(w)}s"] = (
                    float(np.mean(list(buf)[-n:])) if len(buf) >= n else float("nan")
                )
        return out


def history_spec(cfg: dict[str, Any], set_name: str) -> dict[str, Any]:
    spec = cfg["features"]["history_sets"][set_name]
    return {
        "signals": list(spec.get("signals", cfg["features"]["history_signals"])),
        "ewma_tau_s": list(spec["ewma_tau_s"]),
        "rolling_s": list(spec["rolling_s"]),
        "hz": float(cfg["dataset"]["sampling_hz"]),
    }


def build_feature_frame(
    df: pd.DataFrame, cfg: dict[str, Any], set_name: str | None = None
) -> pd.DataFrame:
    """Instantaneous + history features for every profile (full sampling rate)."""
    base = add_instant_features(df)
    spec = history_spec(cfg, set_name or cfg["features"]["superset_history_set"])
    return add_history_features(base, cfg["columns"]["group"], **spec)


def feature_list(cfg: dict[str, Any], base: str, history_set: str = "none") -> list[str]:
    """Ordered model-input columns for a feature set name (``basic`` or ``enhanced``)."""
    if base == "basic":
        cols = list(BASIC_SET)
    elif base == "enhanced":
        cols = list(ENHANCED_SET)
    else:
        raise ValueError(f"unknown base feature set {base!r}")
    spec = history_spec(cfg, history_set)
    cols += history_columns(spec["signals"], spec["ewma_tau_s"], spec["rolling_s"])
    assert_no_forbidden(cols, cfg)
    return cols


def assert_no_forbidden(columns: Iterable[str], cfg: dict[str, Any]) -> None:
    """Fail loudly if a leakage column (measured internal temperature, group id) is an input."""
    forbidden = set(cfg["columns"]["forbidden_features"])
    bad = sorted(c for c in columns if c in forbidden or c.startswith(("stator_", "pm")))
    if bad:
        raise ValueError(f"forbidden leakage features requested: {bad}")
