import math

import numpy as np
import pandas as pd
import pytest

from motor_thermal import features as F
from motor_thermal.inference import instant_features_scalar

SPEC = {
    "signals": ["i_sq", "abs_speed"],
    "ewma_tau_s": [30.0, 120.0],
    "rolling_s": [20.0],
    "hz": 2.0,
}


def test_instant_feature_values():
    df = pd.DataFrame(
        {
            "i_d": [-3.0],
            "i_q": [4.0],
            "u_d": [-2.0],
            "u_q": [10.0],
            "motor_speed": [-600.0],
            "torque": [20.0],
            "ambient": [25.0],
            "coolant": [40.0],
        }
    )
    out = F.add_instant_features(df).iloc[0]
    assert out["i_sq"] == pytest.approx(25.0)
    assert out["i_mag"] == pytest.approx(5.0)
    assert out["u_mag"] == pytest.approx(math.hypot(2, 10))
    assert out["abs_speed"] == 600.0
    assert out["p_mech_w"] == pytest.approx(20.0 * -600.0 * 2 * math.pi / 60, rel=1e-6)
    assert out["p_elec_proxy"] == pytest.approx(1.5 * (6.0 + 40.0))
    assert out["ambient_minus_coolant"] == pytest.approx(-15.0)


def test_scalar_and_vector_instant_features_agree(toy):
    sub = toy.iloc[:50]
    vec = F.add_instant_features(sub)
    for i in (0, 10, 49):
        sc = instant_features_scalar(sub.iloc[i].to_dict())
        for k in F.INSTANT_ENHANCED:
            assert sc[k] == pytest.approx(float(vec.iloc[i][k]), rel=1e-4, abs=1e-3), k


def test_feature_sets_never_contain_forbidden_columns(cfg):
    for base in ("basic", "enhanced"):
        for hist in cfg["features"]["history_sets"]:
            cols = F.feature_list(cfg, base, hist)
            for bad in cfg["columns"]["forbidden_features"]:
                assert bad not in cols
            assert not any(c.startswith(("stator_", "pm")) for c in cols)


@pytest.mark.parametrize(
    "bad", ["stator_winding", "stator_tooth", "stator_yoke", "pm", "profile_id"]
)
def test_forbidden_guard_raises(cfg, bad):
    with pytest.raises(ValueError, match="forbidden"):
        F.assert_no_forbidden(["i_d", bad], cfg)


def test_history_columns_derive_from_inputs_only(cfg):
    """History features are built from drive signals, never from a temperature target."""
    for name, spec in cfg["features"]["history_sets"].items():
        sigs = spec.get("signals", cfg["features"]["history_signals"])
        assert not set(sigs) & set(cfg["columns"]["forbidden_features"]), name


def _frame(toy):
    return F.add_instant_features(toy[toy.profile_id <= 3].reset_index(drop=True))


def test_history_is_strictly_causal(toy):
    """Changing FUTURE samples must not change features at earlier times."""
    base = _frame(toy)
    h1 = F.add_history_features(base, "profile_id", **SPEC)
    k = 1000
    mutated = base.copy()
    mutated.loc[k:, ["i_sq", "abs_speed"]] *= 5.0
    h2 = F.add_history_features(mutated, "profile_id", **SPEC)
    cols = F.history_columns(SPEC["signals"], SPEC["ewma_tau_s"], SPEC["rolling_s"])
    pd.testing.assert_frame_equal(h1.loc[: k - 1, cols], h2.loc[: k - 1, cols])
    assert not np.allclose(
        h1.loc[k + 5 :, cols].to_numpy(), h2.loc[k + 5 :, cols].to_numpy(), equal_nan=True
    )


def test_history_does_not_cross_profile_boundaries(toy):
    base = _frame(toy)
    h = F.add_history_features(base, "profile_id", **SPEC)
    first_rows = base.groupby("profile_id").head(1).index
    for idx in first_rows:
        # EWMA of a single sample is that sample (no carry-over from the previous profile)
        assert h.loc[idx, "ewm_i_sq_30s"] == pytest.approx(base.loc[idx, "i_sq"], rel=1e-5)
        assert h.loc[idx, "ewm_abs_speed_120s"] == pytest.approx(
            base.loc[idx, "abs_speed"], rel=1e-5
        )
        assert np.isnan(h.loc[idx, "roll_i_sq_20s"])
    # profile 2 features are independent of profile 1 data
    alt = base.copy()
    alt.loc[alt.profile_id == 1, ["i_sq", "abs_speed"]] = 1e6
    h_alt = F.add_history_features(alt, "profile_id", **SPEC)
    cols = F.history_columns(SPEC["signals"], SPEC["ewma_tau_s"], SPEC["rolling_s"])
    m = base.profile_id == 2
    pd.testing.assert_frame_equal(h.loc[m, cols], h_alt.loc[m, cols])


def test_cold_start_rolling_is_nan_until_window_full(toy):
    h = F.add_history_features(_frame(toy), "profile_id", **SPEC)
    n = int(20 * 2.0)
    p1 = h[h.profile_id == 1]["roll_i_sq_20s"].to_numpy()
    assert np.isnan(p1[: n - 1]).all()
    assert not np.isnan(p1[n - 1 :]).any()


def test_ewma_matches_closed_form_recursion():
    x = np.array([1.0, 3.0, 2.0, 8.0, 5.0])
    df = pd.DataFrame({"g": 0, "v": x})
    out = F.add_history_features(df, "g", ["v"], [10.0], [], 2.0)
    a = 1 - math.exp(-1 / (10.0 * 2.0))
    num = den = 0.0
    expected = []
    for v in x:
        num, den = (1 - a) * num + v, (1 - a) * den + 1
        expected.append(num / den)
    np.testing.assert_allclose(out["ewm_v_10s"].to_numpy(), expected, rtol=1e-5)


def test_streaming_equals_offline_features(toy):
    """Offline (training-time) and streaming (live) implementations must agree exactly."""
    base = _frame(toy)
    offline = F.add_history_features(base, "profile_id", **SPEC)
    cols = F.history_columns(SPEC["signals"], SPEC["ewma_tau_s"], SPEC["rolling_s"])
    stream = F.StreamingHistory(**SPEC)
    for pid, g in base.groupby("profile_id"):
        stream.reset()
        rows = [stream.update(r) for r in g[["i_sq", "abs_speed"]].to_dict("records")]
        live = pd.DataFrame(rows)[cols]
        np.testing.assert_allclose(
            live.to_numpy(),
            offline.loc[g.index, cols].to_numpy(),
            rtol=2e-4,
            atol=1e-3,
            equal_nan=True,
            err_msg=f"profile {pid}",
        )


def test_streaming_cold_start_is_explicit_nan():
    s = F.StreamingHistory(["v"], [], [5.0], 2.0)
    outs = [s.update({"v": 1.0})["roll_v_5s"] for _ in range(12)]
    assert all(math.isnan(o) for o in outs[:9]) and outs[9] == pytest.approx(1.0)
    s.reset()
    assert math.isnan(s.update({"v": 3.0})["roll_v_5s"])


def test_build_feature_frame_contains_superset(toy, cfg):
    out = F.build_feature_frame(toy, cfg)
    for name in cfg["features"]["history_sets"]:
        assert set(F.feature_list(cfg, "enhanced", name)) <= set(out.columns), name
