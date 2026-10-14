"""Final held-out evaluation, diagnostics, explainability and residual monitoring.

This is the only stage that reads the *test* profiles. It refuses to run unless
``selection.json`` (frozen by the training stage) exists, and it never changes the selection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import features as feat
from . import metrics as met
from . import models as mdl
from . import monitor as mon
from . import plots
from . import splits as spl
from .config import ARTIFACTS_DIR, FIGURES_DIR, config_hash, mode_dir
from .train import load_everything

log = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def feature_group(name: str) -> str:
    """Family label used for grouped permutation importance (correlated features move together)."""
    if name.startswith("ewm_coolant") or name.startswith("roll_coolant"):
        return "coolant history"
    if name.startswith(("ewm_", "roll_")):
        return "load history (i², speed, torque)"
    if name in ("ambient", "coolant", "ambient_minus_coolant"):
        return "ambient / coolant now"
    if name in ("motor_speed", "torque", "abs_speed", "abs_torque"):
        return "speed / torque now"
    if name in ("p_mech_w", "p_elec_proxy", "p_loss_proxy"):
        return "power & loss proxies"
    return "currents / voltages now"


def permutation_importance(
    est: Any,
    columns: list[str],
    frame: pd.DataFrame,
    target: str,
    n_rows: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation importance (MAE increase) on a bounded validation sample, single + grouped."""
    rng = np.random.default_rng(seed)
    sub = frame.sample(n=min(n_rows, len(frame)), random_state=seed)
    x = sub[columns].reset_index(drop=True)
    y = sub[target].to_numpy("float64")
    base = float(np.mean(np.abs(est.predict(x) - y)))

    def score(cols: list[str]) -> list[float]:
        out = []
        for _ in range(repeats):
            xp = x.copy()
            perm = rng.permutation(len(xp))
            for c in cols:
                xp[c] = x[c].to_numpy()[perm]
            out.append(float(np.mean(np.abs(est.predict(xp) - y))) - base)
        return out

    rows = [{"feature": c, "group": feature_group(c), **_ms(score([c]))} for c in columns]
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(feature_group(c), []).append(c)
    grows = [{"group": g, "n_features": len(cs), **_ms(score(cs))} for g, cs in groups.items()]
    single = pd.DataFrame(rows).sort_values("importance_mae_increase", ascending=False)
    grouped = pd.DataFrame(grows).sort_values("importance_mae_increase", ascending=False)
    single.attrs["baseline_val_mae"] = base
    return single, grouped


def _ms(vals: list[float]) -> dict[str, float]:
    return {"importance_mae_increase": float(np.mean(vals)), "std": float(np.std(vals))}


def _elapsed_s(frame: pd.DataFrame, group: str, hz: float) -> np.ndarray:
    return frame.groupby(group).cumcount().to_numpy() / hz


def error_by_elapsed(
    frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, group: str, hz: float
) -> pd.DataFrame:
    edges = [0, 600, 1800, 3600, 7200, 1e9]
    labels = ["0-10 min", "10-30 min", "30-60 min", "1-2 h", ">2 h"]
    t = _elapsed_s(frame, group, hz)
    cut = pd.cut(pd.Series(t), bins=edges, labels=labels, include_lowest=True)
    err = np.abs(p - y)
    out = pd.DataFrame({"elapsed": cut.astype(str), "ae": err}).groupby("elapsed", observed=True)[
        "ae"
    ]
    tab = out.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "mae"}).reset_index()
    tab["elapsed"] = pd.Categorical(tab["elapsed"], labels, ordered=True)
    return tab.sort_values("elapsed").reset_index(drop=True)


def warm_start_test(
    est: Any,
    meta: dict[str, Any],
    test: pd.DataFrame,
    pred_full: np.ndarray,
    cfg: dict[str, Any],
    restart_s: float = 3600.0,
) -> dict[str, Any]:
    """Restart the history buffer mid-profile (motor already warm) and compare with full history."""
    group, target = cfg["columns"]["group"], cfg["columns"]["target"]
    hz = cfg["dataset"]["sampling_hz"]
    spec = meta["history_spec"]
    k0 = int(restart_s * hz)
    full_err, restart_err, early_full, early_restart = [], [], [], []
    used = []
    pos = 0
    for pid, g in test.groupby(group, sort=False):
        n = len(g)
        if n > k0 + int(1800 * hz):
            sl = g.iloc[k0:].copy()
            sl = feat.add_history_features(sl, group, **spec)
            p_new = est.predict(sl[meta["columns"]])
            y = sl[target].to_numpy("float64")
            p_old = pred_full[pos + k0 : pos + n]
            full_err.append(np.abs(p_old - y))
            restart_err.append(np.abs(p_new - y))
            early = int(1800 * hz)
            early_full.append(np.abs(p_old - y)[:early])
            early_restart.append(np.abs(p_new - y)[:early])
            used.append(int(pid))
        pos += n
    if not used:
        return {"profiles_used": [], "note": "no test profile long enough"}
    cat = np.concatenate
    return {
        "restart_after_s": restart_s,
        "profiles_used": used,
        "mae_full_history": float(cat(full_err).mean()),
        "mae_restarted_history": float(cat(restart_err).mean()),
        "mae_first_30min_after_restart_full_history": float(cat(early_full).mean()),
        "mae_first_30min_after_restart_restarted": float(cat(early_restart).mean()),
        "note": "History buffer emptied at restart_after_s while the motor was already warm.",
    }


def run_evaluation(cfg: dict[str, Any], mode: str, csv: str | Path | None = None) -> dict[str, Any]:
    out_dir = mode_dir(mode)
    sel_path = out_dir / "selection.json"
    if not sel_path.is_file():
        raise FileNotFoundError(
            f"{sel_path} missing - run `motor-thermal train --mode {mode}` first"
        )
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    group, target = cfg["columns"]["group"], cfg["columns"]["target"]
    hz = float(cfg["dataset"]["sampling_hz"])
    fig_dir = FIGURES_DIR / mode

    ctx = load_everything(cfg, csv)
    frame, manifest = ctx["frame"], ctx["manifest"]
    saved = spl.load_manifest(ARTIFACTS_DIR / "split_manifest.json")
    for s in spl.SPLITS:
        if saved[s]["profiles"] != manifest[s]["profiles"]:
            raise RuntimeError("split manifest on disk differs from the regenerated one")
    test = spl.split_frame(frame, manifest, "test", group)
    val = spl.split_frame(frame, manifest, "val", group)
    train_all = spl.split_frame(frame, manifest, "train", group)
    y_test = test[target].to_numpy("float64")
    pid_test = test[group].to_numpy()
    # span of the TRAINING inputs (used by the app to warn about extrapolation)
    ranges = {
        c: {
            "min": float(train_all[c].min()),
            "p01": float(train_all[c].quantile(0.01)),
            "p99": float(train_all[c].quantile(0.99)),
            "max": float(train_all[c].max()),
        }
        for c in cfg["columns"]["inputs"]
    }
    dump_json(ranges, out_dir / "input_ranges.json")
    log.info("test profiles %s (%d rows)", manifest["test"]["profiles"], len(test))

    # ---- all candidates, one pass over the untouched test profiles
    val_tab = pd.read_csv(out_dir / "candidates_val.csv")
    index_path = out_dir / "candidates_index.json"
    rows = []
    if index_path.is_file() and (out_dir / "models_all").is_dir():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for cid, info in index.items():
            f = out_dir / "models_all" / info["file"]
            if not f.is_file():
                continue
            est = joblib.load(f)
            p = np.asarray(est.predict(test[info["columns"]]), dtype="float64")
            s = met.summarize_all(pid_test, y_test, p)
            v = val_tab[val_tab.candidate == cid].iloc[0]
            rows.append(
                {
                    "candidate": cid,
                    "track": info["track"],
                    "model": info["model"],
                    "feature_set": info["feature_set"],
                    "n_features": len(info["columns"]),
                    **{f"test_{k}": val_ for k, val_ in s["aggregate"].items() if k != "n"},
                    "test_macro_mae": s["profile_level"]["macro_mae"],
                    "test_worst_profile_mae": s["profile_level"]["worst_profile_mae"],
                    "val_mae": v["val_mae"],
                    "fit_seconds": v["fit_seconds"],
                    "model_size_kb": v["model_size_kb"],
                    "single_row_latency_ms": v["single_row_latency_ms"],
                    "batch_rows_per_s": v["batch_rows_per_s"],
                }
            )
    cand_test = pd.DataFrame(rows)
    if not cand_test.empty:
        cand_test.to_csv(out_dir / "candidates_test.csv", index=False)
        plots.model_comparison(
            cand_test,
            fig_dir / "model_comparison_test.png",
            "test_mae",
            "Test MAE of all candidates (unseen profiles)",
        )

    # ---- winners in detail
    results: dict[str, Any] = {
        "mode": mode,
        "config_hash": config_hash(cfg),
        "selection": selection,
        "test_profiles": manifest["test"]["profiles"],
        "test_rows": int(len(test)),
    }
    preds: dict[str, np.ndarray] = {}
    preds_val: dict[str, np.ndarray] = {}
    metas: dict[str, dict[str, Any]] = {}
    ests: dict[str, Any] = {}
    speed_edges = met.quantile_edges(train_all["abs_speed"].to_numpy())
    torque_edges = met.quantile_edges(train_all["abs_torque"].to_numpy())
    pp_all: dict[str, pd.DataFrame] = {}
    temp_tabs: dict[str, pd.DataFrame] = {}
    for track in ("A", "B"):
        est, meta = mdl.load_model_bundle(out_dir / "models", f"track_{track.lower()}")
        cols = meta["columns"]
        feat.assert_no_forbidden(cols, cfg)
        p = np.asarray(est.predict(test[cols]), dtype="float64")
        preds[track] = p
        preds_val[track] = np.asarray(est.predict(val[cols]), dtype="float64")
        metas[track], ests[track] = meta, est
        s = met.summarize_all(pid_test, y_test, p)
        pp = met.per_profile_metrics(pid_test, y_test, p)
        pp_all[track] = pp
        temp_tab = met.error_by_bins(
            y_test, y_test, p, cfg["evaluation"]["temp_bins"], "temp_range"
        )
        temp_tabs[track] = temp_tab
        region = met.operating_region_table(
            test["abs_speed"].to_numpy(),
            test["abs_torque"].to_numpy(),
            y_test,
            p,
            speed_edges,
            torque_edges,
        )
        elapsed = error_by_elapsed(test, y_test, p, group, hz)
        bench = mdl.benchmark_inference(est, test, cols)
        results[f"track_{track}"] = {
            "candidate": meta["candidate"],
            "n_features": len(cols),
            "columns": cols,
            "test": s,
            "per_profile": pp.reset_index().to_dict("records"),
            "by_temperature_range": temp_tab.to_dict("records"),
            "by_operating_region": region.to_dict("records"),
            "by_elapsed_time_in_profile": elapsed.astype({"elapsed": str}).to_dict("records"),
            "region_edges": {"abs_speed_rpm": speed_edges, "abs_torque_nm": torque_edges},
            "inference": bench,
            "validation_metrics": meta["validation_metrics"],
        }
        pp.to_csv(out_dir / f"per_profile_test_track_{track.lower()}.csv")
        region.to_csv(out_dir / f"operating_region_test_track_{track.lower()}.csv", index=False)
        log.info(
            "[Track %s] TEST MAE %.3f RMSE %.3f R2 %.4f | worst profile %.2f",
            track,
            s["aggregate"]["mae"],
            s["aggregate"]["rmse"],
            s["aggregate"]["r2"],
            s["profile_level"]["worst_profile_mae"],
        )

    # baselines on test for headline comparison (trivial)
    mean_pred = np.full_like(y_test, train_all[target].mean())
    results["train_mean_baseline_test"] = met.summarize_all(pid_test, y_test, mean_pred)

    # ---- predictions (local, ignored by git)
    pred_df = pd.DataFrame(
        {
            group: pid_test,
            "t_s": _elapsed_s(test, group, hz),
            "measured": y_test,
            "pred_A": preds["A"],
            "pred_B": preds["B"],
        }
    )
    (out_dir / "predictions").mkdir(exist_ok=True)
    pred_df.to_parquet(out_dir / "predictions" / "test_predictions.parquet", index=False)
    pd.DataFrame(
        {
            group: val[group].to_numpy(),
            "t_s": _elapsed_s(val, group, hz),
            "measured": val[target].to_numpy("float64"),
            "pred_A": preds_val["A"],
            "pred_B": preds_val["B"],
        }
    ).to_parquet(out_dir / "predictions" / "val_predictions.parquet", index=False)

    # ---- figures: scatter, residuals, per-profile, temp ranges
    plots.pred_vs_actual(
        y_test,
        preds["B"],
        "Track B (history-aware): test",
        fig_dir / "scatter_track_b.png",
        plots.C_B,
    )
    plots.pred_vs_actual(
        y_test,
        preds["A"],
        "Track A (instantaneous): test",
        fig_dir / "scatter_track_a.png",
        plots.C_A,
    )
    plots.residual_diagnostics(y_test, preds["B"], fig_dir / "residuals_track_b.png", "Track B")
    plots.residual_diagnostics(y_test, preds["A"], fig_dir / "residuals_track_a.png", "Track A")
    plots.per_profile_bars(
        {"A instantaneous": pp_all["A"], "B history-aware": pp_all["B"]},
        fig_dir / "per_profile_mae.png",
        "Per-profile MAE on held-out test profiles",
    )
    plots.error_by_range(
        {"A instantaneous": temp_tabs["A"], "B history-aware": temp_tabs["B"]},
        fig_dir / "error_by_temperature_range.png",
    )

    # ---- case studies: typical and worst test profile for Track B
    order = pp_all["B"]["mae"].sort_values()
    worst_pid = int(order.index[-1])
    median_pid = int(order.index[len(order) // 2])
    for tag, pid in (("worst", worst_pid), ("typical", median_pid)):
        m = pid_test == pid
        g = test[m]
        sig = {
            "speed/100 [rpm]": g["motor_speed"].to_numpy() / 100.0,
            "torque [Nm]": g["torque"].to_numpy(),
            "coolant [°C]": g["coolant"].to_numpy(),
            "current |i|/5 [A]": g["i_mag"].to_numpy() / 5.0,
        }
        plots.timeseries(
            pred_df.loc[m, "t_s"].to_numpy() / 60,
            y_test[m],
            {"A instantaneous": preds["A"][m], "B history-aware": preds["B"][m]},
            f"{tag.capitalize()} test profile {pid} (Track B MAE {order[pid]:.2f} °C)",
            fig_dir / f"case_{tag}_profile.png",
            extra=sig,
        )
    results["case_studies"] = {"worst_profile_B": worst_pid, "typical_profile_B": median_pid}

    # ---- explainability (validation sample only)
    imp_cfg = cfg["evaluation"]
    for track in ("A", "B"):
        single, grouped = permutation_importance(
            ests[track],
            metas[track]["columns"],
            val,
            target,
            imp_cfg["permutation_importance_rows"],
            imp_cfg["permutation_repeats"],
            cfg["seed"],
        )
        single.to_csv(out_dir / f"importance_track_{track.lower()}.csv", index=False)
        grouped.to_csv(out_dir / f"importance_groups_track_{track.lower()}.csv", index=False)
        plots.importance_bar(
            single,
            fig_dir / f"importance_track_{track.lower()}.png",
            f"Permutation importance - Track {track} (validation sample)",
        )
        plots.importance_bar(
            grouped.rename(columns={"group": "feature"}),
            fig_dir / f"importance_groups_track_{track.lower()}.png",
            f"Grouped permutation importance - Track {track}",
            top=10,
        )
        results[f"track_{track}"]["importance_top"] = single.head(10).to_dict("records")
        results[f"track_{track}"]["importance_groups"] = grouped.to_dict("records")
        results[f"track_{track}"]["importance_baseline_val_mae"] = single.attrs["baseline_val_mae"]

    # ---- ablation plot (validation)
    abl = pd.read_csv(out_dir / "history_ablation_val.csv")
    plots.ablation_bar(abl, fig_dir / "history_ablation.png")
    results["history_ablation_val"] = abl.to_dict("records")
    # test-side ablation for the same LightGBM-default configurations (reporting only)
    if not cand_test.empty:
        sub = cand_test[cand_test.model.eq("lightgbm") & ~cand_test.candidate.str.contains(r"\[t")]
        results["lightgbm_default_configs_test"] = sub[
            ["candidate", "n_features", "test_mae", "test_rmse", "test_r2", "test_macro_mae"]
        ].to_dict("records")

    # ---- warm-start robustness (Track B)
    results["warm_start_robustness_track_B"] = warm_start_test(
        ests["B"], metas["B"], test, preds["B"], cfg
    )

    # ---- thermal residual monitoring (Track B)
    mcfg = cfg["monitoring"]
    window = int(mcfg["smoothing_window_s"] * hz)
    persist = int(mcfg["persistence_s"] * hz)
    y_val = val[target].to_numpy("float64")
    res_val = y_val - preds_val["B"]
    res_test = y_test - preds["B"]
    sm_val = mon.causal_smooth(res_val, val[group].to_numpy(), window)
    band_smooth = mon.fit_band(sm_val, mcfg["band_quantile"])
    band_raw = float(np.quantile(np.abs(res_val), mcfg["band_quantile"]))
    exc_val = mon.exceedance_report(
        res_val, val[group].to_numpy(), band_raw, band_smooth, window, persist, hz
    )
    exc_test = mon.exceedance_report(res_test, pid_test, band_raw, band_smooth, window, persist, hz)
    inj = mon.injected_offset_sensitivity(
        res_test, pid_test, band_smooth, window, persist, hz, mcfg["injection_offsets_c"]
    )
    results["monitoring"] = {
        "residual_definition": "measured - predicted (Track B)",
        "calibrated_on": "validation profiles",
        "smoothing_window_s": mcfg["smoothing_window_s"],
        "persistence_s": mcfg["persistence_s"],
        "band_quantile": mcfg["band_quantile"],
        "validation": exc_val,
        "test": exc_test,
        "synthetic_injected_offset_on_test": inj,
        "residual_stats_test": {"mean": float(res_test.mean()), "std": float(res_test.std())},
        "caveat": (
            "Healthy held-out profiles only; alarms there are false alarms or model error. "
            "Injected offsets are synthetic and do NOT measure real fault-detection accuracy."
        ),
    }
    for tag, pid in (("worst", worst_pid), ("typical", median_pid)):
        m = pid_test == pid
        sm = mon.causal_smooth(res_test[m], pid_test[m], window)
        fl = mon.persistent_flags(sm, pid_test[m], band_smooth, persist)
        plots.monitoring_plot(
            pred_df.loc[m, "t_s"].to_numpy() / 60,
            y_test[m],
            preds["B"][m],
            sm,
            band_smooth,
            fl,
            fig_dir / f"monitoring_{tag}_profile.png",
            f"Residual monitoring - {tag} test profile {pid}",
        )

    # ---- save
    dump_json(results, out_dir / "metrics_test.json")
    log.info("evaluation written to %s", out_dir)
    return results
