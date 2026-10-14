"""Training stage: candidate comparison, history ablation, tuning, group-CV and model selection.

IMPORTANT: this stage only ever reads the *train* and *validation* profiles. The test profiles are
opened exclusively by :mod:`motor_thermal.evaluate`, after ``selection.json`` has been frozen.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.model_selection import GroupKFold

from . import data as data_mod
from . import features as feat
from . import metrics as met
from . import models as mdl
from . import splits as spl
from .config import ARTIFACTS_DIR, config_hash, mode_dir

log = logging.getLogger(__name__)

PHYSICAL_COLUMNS = ["coolant", "i_sq"]  # steady-state thermal circuit: T ~ T_coolant + R_th * I^2


@dataclass(frozen=True)
class Candidate:
    track: str  # "A" instantaneous / "B" history-aware
    model: str
    feature_set: str  # basic | enhanced | physical | enhanced+<history set>
    columns: tuple[str, ...]
    params: tuple[tuple[str, Any], ...] = ()
    tag: str = ""  # distinguishes tuned trials of the same model / feature set

    @property
    def cid(self) -> str:
        tag = f"[{self.tag}]" if self.tag else ""
        return f"{self.track}/{self.model}{tag}/{self.feature_set}"

    @property
    def param_dict(self) -> dict[str, Any]:
        return dict(self.params)


# ---------------------------------------------------------------- data preparation


def load_everything(cfg: dict[str, Any], csv: str | Path | None) -> dict[str, Any]:
    """Locate/load/validate the dataset, build the split and the full-rate feature frame."""
    csv_path = data_mod.locate_dataset(cfg, csv)
    raw = data_mod.load_measurements(csv_path, cfg)
    report = data_mod.validate_schema(raw, cfg)
    if not report.ok:
        raise ValueError(f"dataset validation failed: {report.errors}")
    for w in report.warnings:
        log.warning("validation warning: %s", w)
    profiles = data_mod.profile_table(raw, cfg)
    manifest = spl.make_split_manifest(profiles, cfg)
    frame = feat.build_feature_frame(raw, cfg)
    return {
        "csv_path": csv_path,
        "raw": raw,
        "frame": frame,
        "profiles": profiles,
        "manifest": manifest,
        "validation": report,
    }


def thin_training_rows(
    frame: pd.DataFrame, manifest: dict[str, Any], cfg: dict[str, Any], mode: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Profile subset (fast mode) and chronological row stride. Features were computed at 2 Hz first."""
    group = cfg["columns"]["group"]
    tcfg = cfg["training"][mode]
    train_ids = list(manifest["train"]["profiles"])
    rng = np.random.default_rng(cfg["seed"])
    frac = float(tcfg["train_profile_fraction"])
    if frac < 1.0:
        n = max(3, round(frac * len(train_ids)))
        chosen = sorted(int(p) for p in rng.choice(train_ids, size=n, replace=False))
    else:
        chosen = train_ids
    sub = frame[frame[group].isin(chosen)]
    stride = int(tcfg["row_stride"])
    pos = sub.groupby(group).cumcount().to_numpy()
    thinned = sub[pos % stride == 0]
    hz = cfg["dataset"]["sampling_hz"]
    info = {
        "mode": mode,
        "train_profiles_used": chosen,
        "n_train_profiles_available": len(train_ids),
        "row_stride": stride,
        "effective_sampling_interval_s": stride / hz,
        "rows_before_thinning": int(len(sub)),
        "rows_after_thinning": int(len(thinned)),
        "kept_fraction_of_all_train_rows": round(len(thinned) / manifest["train"]["rows"], 5),
        "note": (
            "Causal history features (EWMA/rolling) were computed at the native 2 Hz rate BEFORE "
            "thinning, so windows keep their physical duration; only the training rows are thinned."
        ),
    }
    return thinned, info


# ---------------------------------------------------------------- candidates


def _cols(cfg: dict[str, Any], base: str, hist: str = "none") -> tuple[str, ...]:
    return tuple(feat.feature_list(cfg, base, hist))


def track_a_candidates(cfg: dict[str, Any]) -> list[Candidate]:
    basic, enh = _cols(cfg, "basic"), _cols(cfg, "enhanced")
    cands = [
        Candidate("A", "mean_baseline", "basic", basic),
        Candidate("A", "linear", "physical", tuple(PHYSICAL_COLUMNS)),
    ]
    for model in ("linear", "ridge", "random_forest", "hist_gb", "lightgbm"):
        cands.append(Candidate("A", model, "basic", basic))
        cands.append(Candidate("A", model, "enhanced", enh))
    return cands


def _clean_columns(df: pd.DataFrame, cols: list[str]) -> None:
    bad = [c for c in cols if c not in df.columns]
    if bad:
        raise KeyError(f"columns not in frame: {bad}")


def _evaluate_rows(
    rec: mdl.FitRecord, frame: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    group, target = cfg["columns"]["group"], cfg["columns"]["target"]
    pred = mdl.predict(rec, frame, rec.columns)
    summary = met.summarize_all(frame[group].to_numpy(), frame[target].to_numpy(), pred)
    return summary, pred


def _fit_eval(
    cand: Candidate,
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: dict[str, Any],
    mode: str,
    rf_rows: int | None = None,
) -> tuple[mdl.FitRecord, dict[str, Any]]:
    target = cfg["columns"]["target"]
    seed = cfg["seed"]
    n_jobs = cfg["training"]["n_jobs"]
    cols = list(cand.columns)
    _clean_columns(train, cols)
    tr = train
    if cand.model == "random_forest" and rf_rows and len(train) > rf_rows:
        tr = train.sample(n=rf_rows, random_state=seed)
    rec = mdl.fit_candidate(
        cand.model,
        cand.feature_set,
        cols,
        tr,
        target,
        val,
        seed,
        n_jobs,
        cand.param_dict,
        early_stopping_rounds=cfg["training"]["early_stopping_rounds"],
    )
    summary, _ = _evaluate_rows(rec, val, cfg)
    # timing & size on the validation frame (never test)
    bench = mdl.benchmark_inference(rec.estimator, val, cols)
    size_kb = mdl.model_size_kb(rec.estimator, ARTIFACTS_DIR / "_tmp_size.joblib")
    row = {
        "candidate": cand.cid,
        "track": cand.track,
        "model": cand.model,
        "feature_set": cand.feature_set,
        "n_features": len(cols),
        "train_rows": rec.train_rows,
        "fit_seconds": round(rec.fit_seconds, 2),
        "best_iteration": rec.best_iteration,
        "model_size_kb": round(size_kb, 1),
        **{f"val_{k}": v for k, v in summary["aggregate"].items() if k != "n"},
        "val_macro_mae": summary["profile_level"]["macro_mae"],
        "val_worst_profile_mae": summary["profile_level"]["worst_profile_mae"],
        **bench,
    }
    log.info(
        "[%s] val MAE %.3f RMSE %.3f R2 %.4f fit %.1fs",
        cand.cid,
        row["val_mae"],
        row["val_rmse"],
        row["val_r2"],
        rec.fit_seconds,
    )
    return rec, row


def seed_val_maes(
    cand: Candidate, train: pd.DataFrame, val: pd.DataFrame, cfg: dict[str, Any], seeds: list[int]
) -> list[float]:
    """Validation MAE of the same configuration under different random seeds (run-to-run noise)."""
    target = cfg["columns"]["target"]
    y = val[target].to_numpy("float64")
    out = []
    for sd in seeds:
        rec = mdl.fit_candidate(
            cand.model,
            cand.feature_set,
            list(cand.columns),
            train,
            target,
            val,
            sd,
            cfg["training"]["n_jobs"],
            cand.param_dict,
            early_stopping_rounds=cfg["training"]["early_stopping_rounds"],
        )
        out.append(float(np.mean(np.abs(mdl.predict(rec, val, rec.columns) - y))))
    return out


# ---------------------------------------------------------------- tuning & CV


def sample_lgbm_params(space: dict[str, list], n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Trial 0 is the library default configuration; others are seeded random draws."""
    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = [{}]
    seen: set[str] = set()
    while len(trials) < n_trials:
        p = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(p)
    return trials


def group_cv(
    cands: list[tuple[Candidate, int | None]],
    frame: pd.DataFrame,
    profile_ids: list[int],
    cfg: dict[str, Any],
    stride: int,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Group-aware k-fold CV over train+val profiles (test excluded) for fixed configurations."""
    group, target = cfg["columns"]["group"], cfg["columns"]["target"]
    sub = frame[frame[group].isin(profile_ids)]
    pos = sub.groupby(group).cumcount().to_numpy()
    sub = sub[pos % stride == 0].reset_index(drop=True)
    groups = sub[group].to_numpy()
    rows = []
    for cand, n_est in cands:
        fold_mae = []
        params = cand.param_dict
        if cand.model == "lightgbm" and n_est:
            params = {**params, "n_estimators": int(n_est)}
        for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(sub, groups=groups):
            assert not set(groups[tr_idx]) & set(groups[te_idx]), "profile leaked across CV folds"
            est = mdl.make_estimator(cand.model, cfg["seed"], cfg["training"]["n_jobs"], params)
            cols = list(cand.columns)
            est.fit(sub.iloc[tr_idx][cols], sub.iloc[tr_idx][target].to_numpy("float64"))
            pred = est.predict(sub.iloc[te_idx][cols])
            fold_mae.append(
                float(np.mean(np.abs(pred - sub.iloc[te_idx][target].to_numpy("float64"))))
            )
        rows.append(
            {
                "candidate": cand.cid,
                "cv_folds": n_splits,
                "cv_mae_mean": float(np.mean(fold_mae)),
                "cv_mae_std": float(np.std(fold_mae)),
                "cv_mae_folds": [round(x, 3) for x in fold_mae],
            }
        )
        log.info(
            "[CV %s] MAE %.3f +- %.3f", cand.cid, rows[-1]["cv_mae_mean"], rows[-1]["cv_mae_std"]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- environment record


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }


def _meta_for(
    cand: Candidate,
    rec: mdl.FitRecord,
    cfg: dict[str, Any],
    hist_set: str | None,
    mode: str,
    val_row: dict[str, Any],
    provenance: dict[str, Any],
    thin: dict[str, Any],
) -> dict[str, Any]:
    spec = feat.history_spec(cfg, hist_set) if hist_set and hist_set != "none" else None
    return {
        "candidate": cand.cid,
        "track": cand.track,
        "model": cand.model,
        "feature_set": cand.feature_set,
        "mode": mode,
        "columns": rec.columns,
        "history_set": hist_set,
        "history_spec": spec,
        "warmup_seconds": max(spec["rolling_s"], default=0) if spec else 0,
        "instantaneous_feature_kinds": feat.FEATURE_KIND,
        "params": cand.param_dict,
        "best_iteration": rec.best_iteration,
        "target": cfg["columns"]["target"],
        "seed": cfg["seed"],
        "config_hash": config_hash(cfg),
        "train": thin,
        "validation_metrics": {k: v for k, v in val_row.items() if k.startswith("val_")},
        "dataset": {k: provenance[k] for k in ("source", "license", "sha256", "rows", "profiles")},
        "environment": environment_info(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": "Statistical estimate of winding temperature; not a certified protection device.",
    }


# ---------------------------------------------------------------- main entry


def run_training(cfg: dict[str, Any], mode: str, csv: str | Path | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir = mode_dir(mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["training"][mode]
    group = cfg["columns"]["group"]

    ctx = load_everything(cfg, csv)
    frame, manifest = ctx["frame"], ctx["manifest"]
    provenance = data_mod.build_provenance(ctx["csv_path"], ctx["raw"], cfg)
    spl.save_manifest(manifest, ARTIFACTS_DIR / "split_manifest.json")
    (ARTIFACTS_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    ctx["profiles"].to_csv(ARTIFACTS_DIR / "profile_table.csv")

    train, thin = thin_training_rows(frame, manifest, cfg, mode)
    val = spl.split_frame(frame, manifest, "val", group)
    log.info(
        "train rows %d (profiles %d) | val rows %d",
        len(train),
        len(thin["train_profiles_used"]),
        len(val),
    )

    all_dir = out_dir / "models_all"
    shutil.rmtree(all_dir, ignore_errors=True)  # never mix candidates from an earlier run
    rows: list[dict[str, Any]] = []
    records: dict[str, mdl.FitRecord] = {}

    def run(cand: Candidate, **kw) -> dict[str, Any]:
        rec, row = _fit_eval(cand, train, val, cfg, mode, **kw)
        records[cand.cid] = rec
        rows.append(row)
        return row

    # ---- Track A: baselines and candidates, basic vs enhanced features
    for cand in track_a_candidates(cfg):
        run(cand, rf_rows=tcfg["rf_rows"])

    # ---- History ablation (LightGBM default params, identical train/val profiles, several seeds)
    n_seeds = int(tcfg.get("ablation_seeds", 1))
    extra_seeds = [cfg["seed"] + 1 + k for k in range(n_seeds - 1)]
    ablation_rows = []
    best_hist, best_hist_mae = "none", None
    for hist in cfg["features"]["history_sets"]:
        if hist == "none":
            cand = next(c for c in track_a_candidates(cfg) if c.cid == "A/lightgbm/enhanced")
            row = next(r for r in rows if r["candidate"] == cand.cid)
        else:
            cand = Candidate("B", "lightgbm", f"enhanced+{hist}", _cols(cfg, "enhanced", hist))
            row = run(cand)
        maes = [row["val_mae"], *seed_val_maes(cand, train, val, cfg, extra_seeds)]
        ablation_rows.append(
            {
                "history_set": hist,
                "n_features": row["n_features"],
                "val_mae": row["val_mae"],
                "val_mae_seed_mean": float(np.mean(maes)),
                "val_mae_seed_std": float(np.std(maes)),
                "n_seeds": len(maes),
                "val_rmse": row["val_rmse"],
                "val_r2": row["val_r2"],
                "val_macro_mae": row["val_macro_mae"],
                "val_worst_profile_mae": row["val_worst_profile_mae"],
            }
        )
        if hist != "none" and (best_hist_mae is None or np.mean(maes) < best_hist_mae):
            best_hist, best_hist_mae = hist, float(np.mean(maes))
    log.info("best history set on validation: %s (mean MAE %.3f)", best_hist, best_hist_mae)

    # ---- HistGB on the winning history set (NaN handled natively)
    run(Candidate("B", "hist_gb", f"enhanced+{best_hist}", _cols(cfg, "enhanced", best_hist)))

    # ---- Small random search for LightGBM in both tracks
    trials = sample_lgbm_params(cfg["lgbm_search_space"], tcfg["lgbm_trials"], cfg["seed"])
    search_log = []
    for track, hist in (("A", "none"), ("B", best_hist)):
        fs = "enhanced" if hist == "none" else f"enhanced+{hist}"
        for i, p in enumerate(trials):
            if i == 0:
                continue  # trial 0 (defaults) already evaluated above
            p = {**p, "n_estimators": tcfg["lgbm_max_rounds"]}
            cand = Candidate(
                track,
                "lightgbm",
                fs,
                _cols(cfg, "enhanced", hist),
                tuple(sorted(p.items())),
                tag=f"t{i}",
            )
            row = run(cand)
            search_log.append(
                {
                    "track": track,
                    "trial": i,
                    "params": p,
                    "val_mae": row["val_mae"],
                    "best_iteration": row["best_iteration"],
                }
            )

    # ---- Group-aware CV on train+val profiles for the headline configurations
    cv_df = pd.DataFrame()
    if mode == "full":
        tab = pd.DataFrame(rows)
        picks = []
        for track in ("A", "B"):
            sub = tab[(tab.track == track) & (tab.model == "lightgbm")].sort_values("val_mae")
            best = sub.iloc[0]
            cid = best["candidate"]
            rec = records[cid]
            cand = Candidate(
                track,
                "lightgbm",
                best["feature_set"],
                tuple(rec.columns),
                tuple(sorted(rec.params.items())),
                tag=_tag_of(cid),
            )
            picks.append((cand, rec.best_iteration))
        ridge_cols = _cols(cfg, "enhanced")
        picks.append((Candidate("A", "ridge", "enhanced", ridge_cols), None))
        trainval = manifest["train"]["profiles"] + manifest["val"]["profiles"]
        cv_df = group_cv(picks, frame, trainval, cfg, stride=max(4, tcfg["row_stride"]))

    # ---- Frozen model selection (validation only)
    tab = pd.DataFrame(rows)
    winners: dict[str, str] = {}
    for track in ("A", "B"):
        sub = tab[(tab.track == track) & (~tab.model.isin(["mean_baseline"]))].sort_values(
            "val_mae"
        )
        winners[track] = sub.iloc[0]["candidate"]
    selection = {
        "rule": "lowest validation MAE within each track (ties broken by lower fit time)",
        "decided_on": "validation profiles only; test profiles not read in this stage",
        "winners": winners,
        "best_history_set": best_hist,
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ---- Persist all candidates (local, ignored by git) + winners (committed)
    index = {}
    for cid, rec in records.items():
        joblib.dump(rec.estimator, _ensure(all_dir) / f"{safe_name(cid)}.joblib", compress=3)
        index[cid] = {
            "columns": rec.columns,
            "feature_set": rec.feature_set,
            "model": rec.name,
            "track": cid.split("/")[0],
            "params": rec.params,
            "file": f"{safe_name(cid)}.joblib",
        }
    (out_dir / "candidates_index.json").write_text(
        json.dumps(index, indent=2, default=str), encoding="utf-8"
    )
    for track, cid in winners.items():
        rec = records[cid]
        cand_row = tab[tab.candidate == cid].iloc[0].to_dict()
        hist = None
        if "+" in rec.feature_set:
            hist = rec.feature_set.split("+", 1)[1]
        cand = Candidate(
            track,
            rec.name,
            rec.feature_set,
            tuple(rec.columns),
            tuple(sorted(rec.params.items())),
            tag=_tag_of(cid),
        )
        assert cand.cid == cid, (cand.cid, cid)
        meta = _meta_for(cand, rec, cfg, hist, mode, cand_row, provenance, thin)
        mdl.save_model_bundle(out_dir / "models", f"track_{track.lower()}", rec.estimator, meta)

    # ---- Tables & run record
    tab.to_csv(out_dir / "candidates_val.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "history_ablation_val.csv", index=False)
    (out_dir / "lgbm_search_log.json").write_text(
        json.dumps(search_log, indent=2), encoding="utf-8"
    )
    if not cv_df.empty:
        cv_df.to_csv(out_dir / "group_cv.csv", index=False)
    run_record = {
        "mode": mode,
        "seed": cfg["seed"],
        "config_hash": config_hash(cfg),
        "training_subset": thin,
        "run_seconds": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(psutil.Process().memory_info().rss / 1e6, 0),
        "environment": environment_info(),
        "winners": winners,
        "best_history_set": best_hist,
        "split": {k: manifest[k] for k in spl.SPLITS},
        "validation_warnings": ctx["validation"].warnings,
        "validation_facts": ctx["validation"].facts,
    }
    (out_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, default=str), encoding="utf-8"
    )
    log.info("training finished in %.1fs; winners: %s", run_record["run_seconds"], winners)
    return run_record


def _tag_of(cid: str) -> str:
    return cid.split("[", 1)[1].split("]", 1)[0] if "[" in cid else ""


def safe_name(cid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", cid.replace("/", "__"))


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
