import hashlib

import pandas as pd
import pytest

from motor_thermal import data as D
from motor_thermal.config import config_hash, load_config, mode_dir


def test_config_loads_and_hash_is_stable(cfg):
    assert config_hash(cfg) == config_hash(load_config())
    assert cfg["columns"]["target"] == "stator_winding"


def test_config_changes_alter_hash(cfg):
    other = {**cfg, "seed": cfg["seed"] + 1}
    assert config_hash(other) != config_hash(cfg)


def test_inputs_and_forbidden_features_are_disjoint(cfg):
    assert not set(cfg["columns"]["inputs"]) & set(cfg["columns"]["forbidden_features"])
    for temp in ("stator_winding", "stator_tooth", "stator_yoke", "pm", "profile_id"):
        assert temp in cfg["columns"]["forbidden_features"]


def test_mode_dir_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mode_dir("turbo")


def test_validate_schema_accepts_toy_data(toy, cfg):
    rep = D.validate_schema(toy, cfg, strict_counts=False)
    assert rep.ok, rep.errors
    assert rep.facts["profile_blocks"] == rep.facts["profiles"]


def test_validate_schema_reports_missing_column(toy, cfg):
    rep = D.validate_schema(toy.drop(columns=["i_q"]), cfg, strict_counts=False)
    assert not rep.ok and "i_q" in rep.errors[0]


def test_validate_schema_reports_nan(toy, cfg):
    bad = toy.copy()
    bad.loc[5, "torque"] = float("nan")
    assert not D.validate_schema(bad, cfg, strict_counts=False).ok


def test_validate_schema_detects_interleaved_profiles(toy, cfg):
    """Chronology within a profile is only justified if its rows are one contiguous block."""
    shuffled = toy.sample(frac=1.0, random_state=1).reset_index(drop=True)
    rep = D.validate_schema(shuffled, cfg, strict_counts=False)
    assert not rep.ok and any("contiguous" in e for e in rep.errors)


def test_profile_table_counts(toy, cfg):
    tab = D.profile_table(toy, cfg)
    assert len(tab) == toy["profile_id"].nunique()
    assert tab["rows"].sum() == len(toy)
    assert (tab["duration_h"] > 0).all()


def test_locate_dataset_manual_path_and_error(tmp_path, cfg):
    f = tmp_path / "measures_v2.csv"
    f.write_text("a,b\n1,2\n")
    assert D.locate_dataset(cfg, f) == f
    with pytest.raises(D.DatasetUnavailableError):
        D.locate_dataset(cfg, tmp_path / "nope.csv")


def test_load_measurements_dtypes_and_order(toy, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DATA_DIR", tmp_path)
    csv = tmp_path / "t.csv"
    toy.to_csv(csv, index=False)
    df = D.load_measurements(csv, cfg)
    assert str(df["profile_id"].dtype) == "int16"
    assert str(df["torque"].dtype) == "float32"
    pd.testing.assert_series_equal(df["profile_id"], toy["profile_id"].astype("int16"))
    # second call must hit the parquet cache and return identical data
    pd.testing.assert_frame_equal(D.load_measurements(csv, cfg), df)


def test_sha256(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert D.sha256_of(f) == hashlib.sha256(b"abc").hexdigest()
