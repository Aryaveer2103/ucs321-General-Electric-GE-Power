import copy

import pytest

from motor_thermal import data as D
from motor_thermal import splits as S


@pytest.fixture()
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["split"].update(chunk_size=4)
    return c


def test_split_is_exclusive_complete_and_nonempty(toy, small_cfg):
    prof = D.profile_table(toy, small_cfg)
    man = S.make_split_manifest(prof, small_cfg)
    sets = {s: set(man[s]["profiles"]) for s in S.SPLITS}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sets["train"] | sets["val"] | sets["test"] == set(prof.index)
    assert all(sets.values())


def test_split_is_deterministic_and_seed_dependent(toy, small_cfg):
    prof = D.profile_table(toy, small_cfg)
    a = S.make_split_manifest(prof, small_cfg)
    b = S.make_split_manifest(prof, small_cfg)
    assert a == b
    other = copy.deepcopy(small_cfg)
    other["split"]["seed"] = 7
    assert S.make_split_manifest(prof, other) != a


def test_split_frame_never_shares_a_profile(toy, small_cfg):
    prof = D.profile_table(toy, small_cfg)
    man = S.make_split_manifest(prof, small_cfg)
    frames = {s: S.split_frame(toy, man, s, "profile_id") for s in S.SPLITS}
    assert sum(len(f) for f in frames.values()) == len(toy)
    assert not set(frames["train"].profile_id) & set(frames["test"].profile_id)
    assert not set(frames["val"].profile_id) & set(frames["test"].profile_id)


def test_manifest_validation_catches_overlap(toy, small_cfg):
    prof = D.profile_table(toy, small_cfg)
    man = S.make_split_manifest(prof, small_cfg)
    man["test"]["profiles"] = [*man["test"]["profiles"], man["train"]["profiles"][0]]
    with pytest.raises(ValueError, match="appears in both"):
        S.assert_manifest_valid(man, list(prof.index))


def test_manifest_validation_catches_missing_profile(toy, small_cfg):
    prof = D.profile_table(toy, small_cfg)
    man = S.make_split_manifest(prof, small_cfg)
    man["train"]["profiles"] = man["train"]["profiles"][1:]
    with pytest.raises(ValueError, match="mismatch"):
        S.assert_manifest_valid(man, list(prof.index))


def test_manifest_roundtrip(toy, small_cfg, tmp_path):
    man = S.make_split_manifest(D.profile_table(toy, small_cfg), small_cfg)
    S.save_manifest(man, tmp_path / "m.json")
    assert S.load_manifest(tmp_path / "m.json") == man


def test_real_config_gives_roughly_70_15_15(cfg):
    """Structure check on a profile table shaped like the real one (69 profiles, synthetic lengths)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    prof = pd.DataFrame(
        {"rows": rng.integers(2000, 40000, 69), "winding_max": rng.uniform(40, 140, 69)},
        index=pd.Index(range(2, 71), name="profile_id"),
    )
    man = S.make_split_manifest(prof, cfg)
    assert (man["train"]["n_profiles"], man["val"]["n_profiles"], man["test"]["n_profiles"]) == (
        49,
        10,
        10,
    )
