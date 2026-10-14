"""Deterministic profile-wise train/validation/test split.

Whole ``profile_id`` groups are assigned to exactly one split. Rows of one operating
profile are never shared between splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def make_split_manifest(profiles: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Stratified-by-thermal-severity split.

    Profiles are sorted by peak winding temperature, cut into chunks of ``chunk_size`` and each
    chunk donates ``val_per_chunk`` / ``test_per_chunk`` profiles (seeded random choice). This keeps
    mild and severe profiles in every split. Peak temperature is only used to *place* whole
    profiles; no sample-level information crosses splits.
    """
    sp = cfg["split"]
    rng = np.random.default_rng(sp["seed"])
    ordered = profiles.sort_values(["winding_max", "rows"]).index.to_list()
    size, n_val, n_test = sp["chunk_size"], sp["val_per_chunk"], sp["test_per_chunk"]
    assign: dict[int, str] = {}
    for start in range(0, len(ordered), size):
        chunk = ordered[start : start + size]
        perm = list(rng.permutation(len(chunk)))
        full = len(chunk) > n_val + n_test
        k_val = n_val if full else 0
        k_test = n_test if full else 0
        for rank, idx in enumerate(perm):
            if rank < k_val:
                assign[chunk[idx]] = "val"
            elif rank < k_val + k_test:
                assign[chunk[idx]] = "test"
            else:
                assign[chunk[idx]] = "train"
    manifest: dict[str, Any] = {
        "seed": sp["seed"],
        "method": "stratified by peak winding temperature, whole profiles only",
        "chunk_size": size,
    }
    total_rows = float(profiles["rows"].sum())
    for name in SPLITS:
        ids = sorted(int(p) for p, s in assign.items() if s == name)
        rows = int(profiles.loc[ids, "rows"].sum())
        manifest[name] = {
            "profiles": ids,
            "n_profiles": len(ids),
            "rows": rows,
            "row_fraction": round(rows / total_rows, 4),
            "profile_fraction": round(len(ids) / len(profiles), 4),
        }
    assert_manifest_valid(manifest, [int(p) for p in profiles.index])
    return manifest


def assert_manifest_valid(manifest: dict[str, Any], all_profiles: list[int]) -> None:
    """Hard guarantees: disjoint, complete, non-empty."""
    seen: dict[int, str] = {}
    for name in SPLITS:
        ids = manifest[name]["profiles"]
        if not ids:
            raise ValueError(f"split {name!r} is empty")
        for p in ids:
            if p in seen:
                raise ValueError(f"profile {p} appears in both {seen[p]!r} and {name!r}")
            seen[p] = name
    missing = set(all_profiles) - set(seen)
    extra = set(seen) - set(all_profiles)
    if missing or extra:
        raise ValueError(
            f"manifest/profile mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_frame(df: pd.DataFrame, manifest: dict[str, Any], split: str, group: str) -> pd.DataFrame:
    """Rows of ``df`` belonging to ``split`` (row order preserved)."""
    return df[df[group].isin(manifest[split]["profiles"])]
