"""End-to-end smoke test on SYNTHETIC toy data (tiny config), plus model/inference checks.

Verifies that training -> selection -> held-out evaluation -> artifact loading -> app helpers run
and respect the leakage guarantees. The numbers produced here are meaningless for the project.
"""

from __future__ import annotations

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic
from motor_thermal import config as C
from motor_thermal import data as D
from motor_thermal import evaluate as E
from motor_thermal import inference as I  # noqa: E741
from motor_thermal import models as M
from motor_thermal import train as T


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    mp = pytest.MonkeyPatch()
    for mod in (C, T, E):
        mp.setattr(mod, "ARTIFACTS_DIR", tmp / "artifacts", raising=False)
    mp.setattr(E, "FIGURES_DIR", tmp / "figures")
    mp.setattr(D, "DATA_DIR", tmp / "data")
    cfg = copy.deepcopy(C.load_config())
    cfg["split"]["chunk_size"] = 4
    cfg["training"]["fast"].update(
        train_profile_fraction=1.0, row_stride=5, rf_rows=1500, lgbm_trials=2, lgbm_max_rounds=40
    )
    cfg["evaluation"].update(permutation_importance_rows=1500, permutation_repeats=1)
    csv = tmp / "toy.csv"
    make_synthetic(n_profiles=12, n_rows=3200).to_csv(csv, index=False)
    rec = T.run_training(cfg, "fast", csv)
    res = E.run_evaluation(cfg, "fast", csv)
    yield {"cfg": cfg, "tmp": tmp, "rec": rec, "res": res, "csv": csv}
    mp.undo()


def test_training_never_reads_test_profiles(e2e):
    rec = e2e["rec"]
    test_ids = set(rec["split"]["test"]["profiles"])
    assert not test_ids & set(rec["training_subset"]["train_profiles_used"])
    sel = json.loads((e2e["tmp"] / "artifacts" / "fast" / "selection.json").read_text())
    assert "test profiles not read" in sel["decided_on"]


def test_artifacts_written(e2e):
    d = e2e["tmp"] / "artifacts" / "fast"
    for name in (
        "models/track_a.joblib",
        "models/track_b.joblib",
        "models/track_b.json",
        "candidates_val.csv",
        "candidates_test.csv",
        "history_ablation_val.csv",
        "metrics_test.json",
        "run_record.json",
        "input_ranges.json",
    ):
        assert (d / name).is_file(), name
    assert (e2e["tmp"] / "artifacts" / "split_manifest.json").is_file()


def test_saved_models_use_no_forbidden_columns_and_ignore_extra_columns(e2e):
    cfg = e2e["cfg"]
    for track in ("a", "b"):
        meta = json.loads(
            (e2e["tmp"] / "artifacts" / "fast" / "models" / f"track_{track}.json").read_text()
        )
        for bad in cfg["columns"]["forbidden_features"]:
            assert bad not in meta["columns"]
    # predictions are invariant to the measured internal temperatures (target-leakage test)
    sensor = I.VirtualSensor.load("fast", "B")
    raw = pd.read_csv(e2e["csv"]).query("profile_id == 3")
    base = sensor.predict_frame(raw)
    corrupted = raw.copy()
    rng = np.random.default_rng(0)
    for c in ("stator_winding", "stator_tooth", "stator_yoke", "pm"):
        corrupted[c] = rng.normal(500, 50, len(raw))
    np.testing.assert_allclose(sensor.predict_frame(corrupted), base)


def test_prediction_schema(e2e):
    s = I.VirtualSensor.load("fast", "A")
    raw = pd.read_csv(e2e["csv"]).query("profile_id == 2")
    p = s.predict_frame(raw)
    assert p.shape == (len(raw),) and np.isfinite(p).all()
    one = s.predict_one(raw.iloc[10][I.RAW_INPUTS].to_dict())
    assert one == pytest.approx(p[10], rel=1e-4, abs=1e-3)


def test_track_a_rejects_history_but_track_b_streams_like_offline(e2e):
    b = I.VirtualSensor.load("fast", "B")
    with pytest.raises(ValueError):
        b.predict_one({k: 0.0 for k in I.RAW_INPUTS})
    raw = pd.read_csv(e2e["csv"]).query("profile_id == 2").iloc[:700]
    offline = b.predict_frame(raw)
    stream = b.stream()
    live = np.array([stream.update(r) for r in raw[I.RAW_INPUTS].to_dict("records")])
    np.testing.assert_allclose(live, offline, rtol=1e-3, atol=1e-2)  # incl. cold-start NaN rows
    assert np.isfinite(live).all()


def test_candidate_ids_are_unique_and_saved_winner_is_the_validated_one(e2e):
    """Regression: tuned trials once shared an id, so the LAST fitted trial was saved as 'winner'."""
    d = e2e["tmp"] / "artifacts" / "fast"
    tab = pd.read_csv(d / "candidates_val.csv")
    assert tab["candidate"].is_unique
    sel = json.loads((d / "selection.json").read_text())
    ctx = T.load_everything(e2e["cfg"], e2e["csv"])
    val = ctx["frame"][ctx["frame"]["profile_id"].isin(ctx["manifest"]["val"]["profiles"])]
    for track, cid in sel["winners"].items():
        est, meta = M.load_model_bundle(d / "models", f"track_{track.lower()}")
        assert meta["candidate"] == cid
        mae = float(np.mean(np.abs(est.predict(val[meta["columns"]]) - val["stator_winding"])))
        expected = float(tab.loc[tab.candidate == cid, "val_mae"].iloc[0])
        assert mae == pytest.approx(expected, rel=1e-6)
        assert expected == pytest.approx(
            tab[(tab.track == track) & (tab.model != "mean_baseline")]["val_mae"].min()
        )


def test_ablation_reports_seed_spread(e2e):
    abl = pd.read_csv(e2e["tmp"] / "artifacts" / "fast" / "history_ablation_val.csv")
    assert {"val_mae_seed_mean", "val_mae_seed_std", "n_seeds"} <= set(abl.columns)
    assert (abl["n_seeds"] >= 2).all()


def test_metrics_json_has_required_sections(e2e):
    res = e2e["res"]
    for track in ("track_A", "track_B"):
        t = res[track]["test"]
        assert {"mae", "rmse", "r2", "median_ae", "p95_ae"} <= set(t["aggregate"])
        assert "worst_profile_mae" in t["profile_level"]
        assert res[track]["by_temperature_range"] and res[track]["by_operating_region"]
    assert "monitoring" in res and "warm_start_robustness_track_B" in res
    assert set(res["test_profiles"]) == set(e2e["rec"]["split"]["test"]["profiles"])


def test_history_features_help_on_toy_thermal_lag(e2e):
    """On the toy data (built from a first-order lag) the history-aware model must win."""
    res = e2e["res"]
    assert res["track_B"]["test"]["aggregate"]["mae"] < res["track_A"]["test"]["aggregate"]["mae"]


def test_model_artifact_roundtrip_and_missing(e2e, tmp_path):
    est, meta = M.load_model_bundle(e2e["tmp"] / "artifacts" / "fast" / "models", "track_a")
    M.save_model_bundle(tmp_path, "copy", est, meta)
    est2, meta2 = M.load_model_bundle(tmp_path, "copy")
    assert meta2["columns"] == meta["columns"]
    x = pd.DataFrame(np.zeros((2, len(meta["columns"]))), columns=meta["columns"])
    np.testing.assert_allclose(est.predict(x), est2.predict(x))
    with pytest.raises(FileNotFoundError):
        M.load_model_bundle(tmp_path, "nope")


def test_available_modes_and_range_check(e2e):
    assert "fast" in I.available_modes()
    ranges = I.load_input_ranges("fast")
    assert ranges and "coolant" in ranges
    warn = I.check_input_ranges({"coolant": ranges["coolant"]["max"] + 50}, ranges)
    assert warn and "coolant" in warn[0]
    assert I.check_input_ranges({"coolant": ranges["coolant"]["min"]}, ranges) == []


def test_unfitted_estimator_registry_and_unknown_model():
    for name in ("mean_baseline", "linear", "ridge", "random_forest", "hist_gb", "lightgbm"):
        assert M.make_estimator(name, 0, 1) is not None
    with pytest.raises(ValueError):
        M.make_estimator("transformer", 0, 1)


def test_evaluation_refuses_to_run_without_frozen_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ARTIFACTS_DIR", tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="train"):
        E.run_evaluation(C.load_config(), "fast", None)


def test_estimators_are_picklable(e2e):
    est, _ = M.load_model_bundle(e2e["tmp"] / "artifacts" / "fast" / "models", "track_b")
    assert joblib.hash(est)
