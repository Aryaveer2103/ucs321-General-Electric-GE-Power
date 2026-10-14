import numpy as np
import pytest

from motor_thermal import metrics as M
from motor_thermal import monitor as mon


def test_regression_metrics_known_values():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([12.0, 18.0, 33.0, 40.0])
    m = M.regression_metrics(y, p)
    assert m["mae"] == pytest.approx((2 + 2 + 3 + 0) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 0) / 4))
    assert m["r2"] == pytest.approx(1 - 17 / 500)
    assert m["max_ae"] == 3.0
    assert m["n"] == 4


def test_perfect_prediction_and_mean_baseline_r2():
    y = np.linspace(0, 100, 50)
    assert M.regression_metrics(y, y)["mae"] == 0.0
    assert M.regression_metrics(y, np.full_like(y, y.mean()))["r2"] == pytest.approx(0.0, abs=1e-12)


def test_metrics_reject_bad_shapes():
    with pytest.raises(ValueError):
        M.regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_profile_level_differs_from_sample_weighted():
    pid = np.array([1] * 90 + [2] * 10)
    y = np.zeros(100)
    p = np.array([1.0] * 90 + [11.0] * 10)
    agg = M.regression_metrics(y, p)["mae"]
    pp = M.per_profile_metrics(pid, y, p)
    s = M.summarize_profiles(pp)
    assert agg == pytest.approx(2.0)
    assert s["macro_mae"] == pytest.approx(6.0)
    assert s["worst_profile_id"] == 2


def test_error_bins_and_regions():
    y = np.array([10.0, 30.0, 50.0, 70.0])
    p = y + np.array([1.0, 2.0, 3.0, 4.0])
    t = M.error_by_bins(y, y, p, [0, 20, 60, 100], "rng")
    assert list(t["n"]) == [1, 2, 1]
    edges = M.quantile_edges(np.arange(90.0))
    r = M.operating_region_table(
        np.arange(90.0), np.arange(90.0), np.zeros(90), np.ones(90), edges, edges
    )
    assert r["n"].sum() == 90


def test_causal_smooth_nan_start_and_no_cross_profile_mix():
    res = np.r_[np.ones(10), 100 * np.ones(10)]
    pid = np.r_[np.zeros(10), np.ones(10)].astype(int)
    sm = mon.causal_smooth(res, pid, 4)
    assert np.isnan(sm[:3]).all() and sm[3] == 1.0
    assert np.isnan(sm[10:13]).all() and sm[13] == 100.0  # profile 2 not polluted by profile 1


def test_persistent_flags_require_persistence_and_reset():
    sm = np.zeros(30)
    sm[5] = 10  # isolated spike
    sm[10:20] = 10  # persistent deviation
    pid = np.zeros(30, dtype=int)
    f = mon.persistent_flags(sm, pid, band=5.0, persist=5)
    assert not f[5]
    assert not f[10:14].any() and f[14:20].all() and not f[20:].any()


def test_band_calibrated_on_validation_quantile():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 2.0, 100000)
    band = mon.fit_band(v, 0.99)
    assert band == pytest.approx(2.0 * 2.576, rel=0.03)
    with pytest.raises(ValueError):
        mon.fit_band(np.array([np.nan]), 0.99)


def test_injected_offset_is_detected_for_large_bias_only():
    rng = np.random.default_rng(1)
    res = rng.normal(0, 0.5, 6000)
    pid = np.r_[np.zeros(3000), np.ones(3000)].astype(int)
    out = mon.injected_offset_sensitivity(
        res, pid, band_smooth=2.0, window=20, persist=40, hz=2.0, offsets_c=[0.5, 6.0]
    )
    assert out[0]["profiles_alarming"] == 0
    assert out[1]["profiles_alarming"] == 2 and out[1]["profiles_newly_alarming"] == 2


def test_exceedance_report_on_clean_residual_has_no_alarm():
    rng = np.random.default_rng(2)
    res = rng.normal(0, 0.3, 4000)
    pid = np.zeros(4000, dtype=int)
    rep = mon.exceedance_report(
        res, pid, band_raw=3.0, band_smooth=2.0, window=20, persist=40, hz=2.0
    )
    assert rep["persistent_alarm_frac"] == 0.0 and rep["profiles_with_any_alarm"] == 0
