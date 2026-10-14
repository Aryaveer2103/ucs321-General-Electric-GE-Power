"""Dataset acquisition, loading, schema validation and provenance."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR

log = logging.getLogger(__name__)

ENV_CSV = "MOTOR_THERMAL_CSV"


class DatasetUnavailableError(RuntimeError):
    """Raised when the dataset can neither be found locally nor downloaded."""


def find_local_dataset(cfg: dict[str, Any], manual_path: str | Path | None = None) -> Path | None:
    """Look for the CSV on disk only (explicit path, env var, ``data/``, kagglehub cache). No network."""
    file_name = cfg["dataset"]["file_name"]
    if manual_path:  # an explicit path is exclusive: never silently fall back to another file
        return Path(manual_path) if Path(manual_path).is_file() else None
    candidates: list[Path] = []
    if os.environ.get(ENV_CSV):
        candidates.append(Path(os.environ[ENV_CSV]))
    candidates.append(DATA_DIR / file_name)
    handle = cfg["dataset"]["kaggle_handle"]
    hub = Path.home() / ".cache" / "kagglehub" / "datasets" / handle / "versions"
    if hub.is_dir():
        candidates += sorted(hub.glob(f"*/{file_name}"), reverse=True)
    return next((c for c in candidates if c.is_file()), None)


def locate_dataset(cfg: dict[str, Any], manual_path: str | Path | None = None) -> Path:
    """Find ``measures_v2.csv``.

    Order: explicit path -> ``MOTOR_THERMAL_CSV`` env var -> ``data/<file>`` ->
    kagglehub download (public dataset, anonymous access works as of this project).
    """
    file_name = cfg["dataset"]["file_name"]
    found = find_local_dataset(cfg, manual_path)
    if found:
        log.info("Using dataset file %s", found)
        return found
    if manual_path:
        raise DatasetUnavailableError(f"--csv path does not exist: {manual_path}")
    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(cfg["dataset"]["kaggle_handle"]))
    except Exception as exc:  # network / auth / package problems
        raise DatasetUnavailableError(
            "Could not download the dataset with kagglehub "
            f"({type(exc).__name__}: {exc}). Download measures_v2.csv manually from "
            "https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature and either put it "
            f"in data/ or set {ENV_CSV} / pass --csv."
        ) from exc
    path = folder / file_name
    if not path.is_file():
        raise DatasetUnavailableError(f"{file_name} not found in downloaded folder {folder}")
    return path


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_measurements(
    csv_path: str | Path, cfg: dict[str, Any], use_cache: bool = True
) -> pd.DataFrame:
    """Read the CSV (row order preserved) into compact dtypes; cache as parquet."""
    csv_path = Path(csv_path)
    cache = DATA_DIR / "cache" / f"{csv_path.stem}.parquet"
    if use_cache and cache.is_file() and cache.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_parquet(cache)
    df = pd.read_csv(csv_path)
    group = cfg["columns"]["group"]
    for col in df.columns:
        if col == group:
            df[col] = df[col].astype("int16")
        else:
            df[col] = df[col].astype("float32")
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def validate_schema(
    df: pd.DataFrame, cfg: dict[str, Any], strict_counts: bool = True
) -> ValidationReport:
    """Check schema, missing values, plausibility and per-profile row contiguity.

    Row contiguity matters: the file has no timestamp column, so chronology is only
    justified if every ``profile_id`` occupies one unbroken block of rows.
    """
    rep = ValidationReport()
    cols = cfg["columns"]
    group, target = cols["group"], cols["target"]
    required = [*cols["inputs"], target, group]
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.error(f"missing required columns: {missing}")
        return rep

    n_missing = int(df[required].isna().sum().sum())
    rep.facts["missing_values"] = n_missing
    if n_missing:
        rep.error(f"{n_missing} missing values in required columns")

    rep.facts["rows"] = int(len(df))
    rep.facts["profiles"] = int(df[group].nunique())
    if strict_counts:
        exp = cfg["dataset"]
        if len(df) != exp["expected_rows"]:
            rep.warnings.append(f"row count {len(df)} != documented {exp['expected_rows']}")
        if df[group].nunique() != exp["expected_profiles"]:
            rep.warnings.append(
                f"profile count {df[group].nunique()} != documented {exp['expected_profiles']}"
            )

    pid = df[group].to_numpy()
    n_blocks = int((pid[1:] != pid[:-1]).sum() + 1) if len(pid) else 0
    rep.facts["profile_blocks"] = n_blocks
    if n_blocks != df[group].nunique():
        rep.error(
            f"profile rows are not contiguous ({n_blocks} blocks for {df[group].nunique()} "
            "profiles); row order cannot be used as time"
        )

    for col, (lo, hi) in cfg.get("physical_ranges", {}).items():
        if col in df.columns:
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out:
                rep.warnings.append(f"{col}: {n_out} values outside plausible range [{lo}, {hi}]")
    rep.facts["duplicate_rows"] = int(df.duplicated().sum())
    return rep


def profile_table(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per profile: length, duration and thermal summary."""
    cols = cfg["columns"]
    hz = cfg["dataset"]["sampling_hz"]
    g = df.groupby(cols["group"], sort=True)
    out = g.agg(
        rows=(cols["target"], "size"),
        winding_start=(cols["target"], "first"),
        winding_max=(cols["target"], "max"),
        winding_mean=(cols["target"], "mean"),
        speed_abs_mean=("motor_speed", lambda s: float(np.abs(s).mean())),
        coolant_mean=("coolant", "mean"),
    )
    out["duration_h"] = out["rows"] / hz / 3600.0
    return out


def build_provenance(csv_path: Path, df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Provenance record stored alongside every trained model."""
    cols = cfg["columns"]
    hz = cfg["dataset"]["sampling_hz"]
    return {
        "source": f"https://www.kaggle.com/datasets/{cfg['dataset']['kaggle_handle']}",
        "origin": "Paderborn University LEA motor test bench (Kirchgässner, Wallscheid, Böcker)",
        "license": cfg["dataset"]["license"],
        "file_name": csv_path.name,
        "kaggle_cache_version_dir": csv_path.parent.name if "versions" in str(csv_path) else None,
        "file_size_bytes": csv_path.stat().st_size,
        "sha256": sha256_of(csv_path),
        "rows": int(len(df)),
        "profiles": int(df[cols["group"]].nunique()),
        "columns": list(df.columns),
        "assumed_sampling_hz": hz,
        "total_hours_at_assumed_rate": round(len(df) / hz / 3600.0, 2),
        "timestamps_in_file": False,
        "note": "Row order inside each profile_id block is used as time (validated contiguity).",
    }
