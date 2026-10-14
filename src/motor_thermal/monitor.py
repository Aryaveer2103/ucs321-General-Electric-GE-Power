"""Thermal residual monitoring: residual = measured - predicted winding temperature.

This is an *experimental* indicator. A persistent residual can mean a sensor fault, a cooling
change, an operating condition outside the training data, ordinary model error, or a real motor
issue - the method cannot tell these apart, and it is NOT a certified protection function. No
fault-detection accuracy is claimed because the dataset contains no labelled faults.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def causal_smooth(values: np.ndarray, profile_ids: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of the residual within each profile (NaN until the window is full)."""
    out = np.full(len(values), np.nan)
    bounds = np.flatnonzero(np.r_[True, profile_ids[1:] != profile_ids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        out[a:b] = pd.Series(values[a:b]).rolling(window, min_periods=window).mean().to_numpy()
    return out


def persistent_flags(
    smoothed: np.ndarray, profile_ids: np.ndarray, band: float, persist: int
) -> np.ndarray:
    """True once ``|smoothed| > band`` has held for ``persist`` consecutive samples (same profile)."""
    exceed = np.nan_to_num(np.abs(smoothed), nan=0.0) > band
    flags = np.zeros(len(smoothed), dtype=bool)
    bounds = np.flatnonzero(np.r_[True, profile_ids[1:] != profile_ids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        run = 0
        for i in range(a, b):
            run = run + 1 if exceed[i] else 0
            flags[i] = run >= persist
    return flags


def fit_band(smoothed_val: np.ndarray, quantile: float) -> float:
    """Empirical band: ``quantile`` of |smoothed residual| on VALIDATION profiles (not test)."""
    a = np.abs(smoothed_val[~np.isnan(smoothed_val)])
    if a.size == 0:
        raise ValueError("no finite smoothed residuals to calibrate the band")
    return float(np.quantile(a, quantile))


def _events(flags: np.ndarray, profile_ids: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for pid in pd.unique(profile_ids):
        f = flags[profile_ids == pid]
        starts = np.flatnonzero(np.r_[f[0], f[1:] & ~f[:-1]]) if f.size else []
        counts[int(pid)] = int(len(starts))
    return counts


def exceedance_report(
    residual: np.ndarray,
    profile_ids: np.ndarray,
    band_raw: float,
    band_smooth: float,
    window: int,
    persist: int,
    hz: float,
) -> dict[str, Any]:
    """How often a (healthy, held-out) profile trips the band, raw vs smoothed vs persistent."""
    smooth = causal_smooth(residual, profile_ids, window)
    flags = persistent_flags(smooth, profile_ids, band_smooth, persist)
    per_profile = []
    for pid in pd.unique(profile_ids):
        m = profile_ids == pid
        per_profile.append(
            {
                "profile_id": int(pid),
                "raw_exceed_frac": float(np.mean(np.abs(residual[m]) > band_raw)),
                "smooth_exceed_frac": float(np.nanmean(np.abs(smooth[m]) > band_smooth)),
                "persistent_alarm_frac": float(flags[m].mean()),
                "alarm_events": _events(flags[m], profile_ids[m])[int(pid)],
                "duration_h": float(m.sum() / hz / 3600),
            }
        )
    pp = pd.DataFrame(per_profile)
    return {
        "band_raw_c": band_raw,
        "band_smoothed_c": band_smooth,
        "raw_exceed_frac": float(np.mean(np.abs(residual) > band_raw)),
        "smooth_exceed_frac": float(np.nanmean(np.abs(smooth) > band_smooth)),
        "persistent_alarm_frac": float(flags.mean()),
        "profiles_with_any_alarm": int((pp["alarm_events"] > 0).sum()),
        "n_profiles": int(len(pp)),
        "alarm_events_per_hour": float(pp["alarm_events"].sum() / pp["duration_h"].sum()),
        "per_profile": per_profile,
    }


def injected_offset_sensitivity(
    residual: np.ndarray,
    profile_ids: np.ndarray,
    band_smooth: float,
    window: int,
    persist: int,
    hz: float,
    offsets_c: list[float],
) -> list[dict[str, Any]]:
    """SYNTHETIC check: add a constant bias to the measured temperature of held-out profiles.

    A positive bias ``d`` raises the residual by ``d``. Reported: fraction of profiles for which a
    persistent alarm appears and the median detection delay. This illustrates sensitivity of the
    detector to a sensor-offset-like deviation; it is not a validation on real faults.
    """
    base_flags = persistent_flags(
        causal_smooth(residual, profile_ids, window), profile_ids, band_smooth, persist
    )
    ids = list(pd.unique(profile_ids))
    already = {pid: bool(base_flags[profile_ids == pid].any()) for pid in ids}
    rows = []
    for d in offsets_c:
        smooth = causal_smooth(residual + d, profile_ids, window)
        flags = persistent_flags(smooth, profile_ids, band_smooth, persist)
        delays, alarming, newly = [], 0, 0
        for pid in ids:
            f = flags[profile_ids == pid]
            if f.any():
                alarming += 1
                newly += int(not already[pid])
                delays.append(int(np.argmax(f)) / hz)
        rows.append(
            {
                "offset_c": float(d),
                "n_profiles": len(ids),
                "profiles_alarming": alarming,
                "profiles_alarming_without_offset": int(sum(already.values())),
                "profiles_newly_alarming": newly,
                "median_first_alarm_s": float(np.median(delays)) if delays else None,
                "alarm_frac_of_samples": float(flags.mean()),
            }
        )
    return rows
