"""Fill the markdown templates (project report, viva preparation, README results block)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import features as feat
from .config import ARTIFACTS_DIR, PROJECT_ROOT, REPORTS_DIR, mode_dir
from .report import TEMPLATES, md_table, render


def _n_tests() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        m = re.search(r"(\d+) tests? collected", out)
        return m.group(1) if m else "the"
    except Exception:
        return "the"


def _f(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}"


def _stream_us(meta_path: Path, cfg: dict[str, Any]) -> float:
    """Measured cost of one StreamingHistory.update (microseconds) for the Track B recipe."""
    import json
    import time

    spec = json.loads(meta_path.read_text(encoding="utf-8"))["history_spec"]
    st = feat.StreamingHistory(**spec)
    sample = {s: 1.0 for s in spec["signals"]}
    for _ in range(700):  # fill the longest rolling window
        st.update(sample)
    t0 = time.perf_counter()
    n = 3000
    for _ in range(n):
        st.update(sample)
    return (time.perf_counter() - t0) / n * 1e6


def _worst_story(pred_path: Path, pid: int, profiles: pd.DataFrame) -> tuple[str, float, float]:
    """Describe why a profile is hard, from the saved test predictions (local artifact)."""
    row = profiles.loc[pid]
    base = (
        f"It lasts {row['duration_h']:.1f} h, starts at {row['winding_start']:.0f} °C and peaks at "
        f"{row['winding_max']:.0f} °C (coolant mean {row['coolant_mean']:.0f} °C)."
    )
    if not pred_path.is_file():
        return base, float("nan"), float("nan")
    p = pd.read_parquet(pred_path)
    g = p[p.profile_id == pid]
    err = g["pred_B"] - g["measured"]
    early = np.abs(err[g.t_s < 600]).mean()
    late = np.abs(err[g.t_s >= 600]).mean()
    hot = g["measured"] > g["measured"].quantile(0.9)
    bias_hot = err[hot].mean()
    direction = "under-estimates" if bias_hot < 0 else "over-estimates"
    story = (
        f"{base} The model {direction} the hottest 10 % of the profile by {abs(bias_hot):.1f} °C on average; "
        f"the MAE is {early:.1f} °C in the first 10 min and {late:.1f} °C afterwards."
    )
    return story, float(early), float(late)


def build_values(
    cfg: dict[str, Any], mode: str, g: dict[str, Any], sent: dict[str, str]
) -> dict[str, Any]:
    r, run, prov, eda = g["res"], g["run"], g["prov"], g["eda"]
    cand, man = g["cand_test"], g["manifest"]
    A, B = r["track_A"], r["track_B"]
    am, bm = A["test"]["aggregate"], B["test"]["aggregate"]
    sel = r["selection"]["winners"]
    mon = r["monitoring"]
    pred_path = mode_dir(mode) / "predictions" / "test_predictions.parquet"
    sub = run["training_subset"]
    env = run["environment"]
    val = {}
    val.update(
        mode=mode,
        mode_label="full (final results)" if mode == "full" else "fast (preliminary subset)",
        config_hash=run["config_hash"],
        seed=run["seed"],
        rows=f"{prov['rows']:,}",
        hours=_f(prov["total_hours_at_assumed_rate"], 0),
        profiles=prov["profiles"],
        n_train=man["train"]["n_profiles"],
        n_val=man["val"]["n_profiles"],
        n_test=man["test"]["n_profiles"],
        n_tests=_n_tests(),
        A_mae=_f(am["mae"]),
        A_rmse=_f(am["rmse"]),
        A_r2=_f(am["r2"], 3),
        A_p95=_f(am["p95_ae"]),
        A_worst=_f(A["test"]["profile_level"]["worst_profile_mae"]),
        B_mae=_f(bm["mae"]),
        B_rmse=_f(bm["rmse"]),
        B_r2=_f(bm["r2"], 3),
        B_p95=_f(bm["p95_ae"]),
        B_worst=_f(B["test"]["profile_level"]["worst_profile_mae"]),
        ratio=_f(am["mae"] / bm["mae"], 1),
        base_mae=_f(r["train_mean_baseline_test"]["aggregate"]["mae"]),
        win_A=sel["A"],
        win_B=sel["B"],
        B_nfeat=B["n_features"],
    )
    # ---- dataset / EDA
    desc = eda["describe"]
    val.update(
        prov_source=prov["source"],
        prov_origin=prov["origin"],
        prov_license=prov["license"],
        prov_file=prov["file_name"],
        file_mb=_f(prov["file_size_bytes"] / 1e6, 0),
        sha256=prov["sha256"],
        n_cols=len(prov["columns"]),
        columns=", ".join(prov["columns"]),
        prof_min_rows=f"{int(eda['profile_length_rows']['min']):,}",
        prof_max_rows=f"{int(eda['profile_length_rows']['max']):,}",
        prof_med_h=_f(eda["profile_duration_h"]["50%"], 1),
        blocks=eda["validation"]["profile_blocks"],
        step_med=_f(eda["sampling_sanity"]["winding_abs_step_c_median"], 3),
        step_p999=_f(eda["sampling_sanity"]["winding_abs_step_c_p99.9"], 2),
        missing=eda["validation"]["missing_values"],
        dups=eda["validation"]["duplicate_rows"],
        const_runs=eda["constant_and_duplicate_readings"]["winding_constant_runs_ge_10s"],
        const_samples=eda["constant_and_duplicate_readings"]["winding_samples_in_constant_runs"],
        standstill_frac=_f(
            eda["constant_and_duplicate_readings"]["fraction_of_rows_with_|speed|<1rpm"] * 100, 1
        ),
        plausibility_text=(
            "all columns lie inside the physically plausible ranges in `configs/default.yaml` (no warnings)."
            if not eda["validation"]["warnings"]
            else "validation warnings: " + "; ".join(eda["validation"]["warnings"]) + "."
        ),
        wind_min=_f(desc["stator_winding"]["min"], 0),
        wind_max=_f(desc["stator_winding"]["max"], 0),
        cool_min=_f(desc["coolant"]["min"], 0),
        cool_max=_f(desc["coolant"]["max"], 0),
        amb_min=_f(desc["ambient"]["min"], 0),
        amb_max=_f(desc["ambient"]["max"], 0),
        n_cold=eda["start_state"]["profiles_starting_within_5C_of_ambient"],
        n_warm=eda["start_state"]["profiles_starting_warm_(>5C_above_ambient)"],
        max_start=_f(eda["start_state"]["max_start_winding_c"], 0),
        corr_tooth=_f(eda["corr_with_winding_train"]["stator_tooth"]),
        corr_yoke=_f(eda["corr_with_winding_train"]["stator_yoke"]),
        corr_pm=_f(eda["corr_with_winding_train"]["pm"]),
        corr_isq=_f(eda["corr_with_winding_train"]["i_sq"]),
        corr_torque_iq=_f(eda["corr_torque_iq_train"], 3),
    )
    # ---- training description
    val.update(
        es_rounds=cfg["training"]["early_stopping_rounds"],
        chunk=cfg["split"]["chunk_size"],
        trials=cfg["training"][mode]["lgbm_trials"],
        train_profiles_used=len(sub["train_profiles_used"]),
        stride=sub["row_stride"],
        interval=_f(sub["effective_sampling_interval_s"], 1),
        train_rows=f"{sub['rows_after_thinning']:,}",
        train_rows_all=f"{man['train']['rows']:,}",
        kept_pct=_f(sub["kept_fraction_of_all_train_rows"] * 100, 1),
        run_minutes=_f(run["run_seconds"] / 60, 1),
        peak_rss=f"{run['peak_rss_mb']:.0f}",
        cpu=env["cpu_count"],
        ram=env["ram_gb"],
        python=env["python"],
        sklearn=env["scikit_learn"],
        lightgbm=env["lightgbm"],
        test_ids=", ".join(map(str, man["test"]["profiles"])),
        val_ids=", ".join(map(str, man["val"]["profiles"])),
        perm_rows=f"{cfg['evaluation']['permutation_importance_rows']:,}",
        perm_rep=cfg["evaluation"]["permutation_repeats"],
    )
    # ---- tables
    split_df = pd.DataFrame(
        [
            {
                "split": s,
                "profiles": man[s]["n_profiles"],
                "rows": f"{man[s]['rows']:,}",
                "% of rows": f"{man[s]['row_fraction'] * 100:.1f}",
            }
            for s in ("train", "val", "test")
        ]
    )
    val["table_splits"] = md_table(split_df)
    val["table_feature_kinds"] = "\n".join(f"| `{k}` | {v} |" for k, v in feat.FEATURE_KIND.items())
    cv = g["cv"]
    if cv is not None:
        cvt = cv.copy()
        cvt["MAE mean ± std [°C]"] = [
            f"{a:.2f} ± {b:.2f}" for a, b in zip(cvt.cv_mae_mean, cvt.cv_mae_std, strict=True)
        ]
        cvt["folds"] = cvt["cv_mae_folds"]
        val["table_cv"] = md_table(cvt[["candidate", "MAE mean ± std [°C]", "folds"]])
        get = lambda pat: cvt[cvt.candidate.str.startswith(pat)].iloc[0]  # noqa: E731
        val.update(
            cv_A_mean=_f(get("A/lightgbm").cv_mae_mean),
            cv_A_std=_f(get("A/lightgbm").cv_mae_std),
            cv_B_mean=_f(get("B/lightgbm").cv_mae_mean),
            cv_B_std=_f(get("B/lightgbm").cv_mae_std),
            cv_ridge_mean=_f(get("A/ridge").cv_mae_mean),
        )
    else:
        val["table_cv"] = "*(group-aware CV is run in full mode only)*"
        for k in ("cv_A_mean", "cv_A_std", "cv_B_mean", "cv_B_std", "cv_ridge_mean"):
            val[k] = "n/a"

    shown = cand[~cand.candidate.str.contains(r"\[t") | cand.candidate.isin(sel.values())].copy()
    shown = shown.sort_values(["track", "test_mae"], ascending=[True, False])
    shown = shown.sort_values(["track", "test_mae"])
    tab = shown[
        [
            "candidate",
            "n_features",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_p95_ae",
            "test_macro_mae",
            "test_worst_profile_mae",
            "val_mae",
            "fit_seconds",
            "model_size_kb",
        ]
    ].rename(
        columns={
            "test_mae": "test MAE",
            "test_rmse": "RMSE",
            "test_r2": "R²",
            "test_p95_ae": "p95 AE",
            "test_macro_mae": "profile-mean MAE",
            "test_worst_profile_mae": "worst-profile MAE",
            "val_mae": "val MAE",
            "fit_seconds": "fit [s]",
            "model_size_kb": "size [kB]",
        }
    )
    tab["R²"] = tab["R²"].map(lambda v: f"{v:.3f}")
    val["table_candidates_test"] = md_table(tab)
    head = pd.DataFrame(
        {
            "metric": [
                "MAE [°C]",
                "RMSE [°C]",
                "R²",
                "median abs. error",
                "95th pct abs. error",
                "bias (est − meas)",
                "profile-mean MAE",
                "median-profile MAE",
                "worst-profile MAE",
            ],
            "Track A": [
                am["mae"],
                am["rmse"],
                am["r2"],
                am["median_ae"],
                am["p95_ae"],
                am["bias"],
                A["test"]["profile_level"]["macro_mae"],
                A["test"]["profile_level"]["median_profile_mae"],
                A["test"]["profile_level"]["worst_profile_mae"],
            ],
            "Track B": [
                bm["mae"],
                bm["rmse"],
                bm["r2"],
                bm["median_ae"],
                bm["p95_ae"],
                bm["bias"],
                B["test"]["profile_level"]["macro_mae"],
                B["test"]["profile_level"]["median_profile_mae"],
                B["test"]["profile_level"]["worst_profile_mae"],
            ],
        }
    )
    val["table_headline"] = md_table(head, "{:.3f}")
    ab = g["ablation"]
    abt = pd.DataFrame(
        {
            "history set": ab["history_set"],
            "features": ab["n_features"],
            "val MAE (mean ± std over seeds)": [
                f"{m:.2f} ± {s:.2f}"
                for m, s in zip(ab.val_mae_seed_mean, ab.val_mae_seed_std, strict=True)
            ],
            "val RMSE": ab["val_rmse"],
            "worst-profile val MAE": ab["val_worst_profile_mae"],
        }
    )
    val["table_ablation"] = md_table(abt)
    val["ablation_sentence"] = sent["ablation_sentence"]
    dflt = pd.DataFrame(r.get("lightgbm_default_configs_test", []))
    val["table_lgbm_default_test"] = (
        md_table(
            dflt.rename(
                columns={
                    "test_mae": "test MAE",
                    "test_rmse": "RMSE",
                    "test_r2": "R²",
                    "test_macro_mae": "profile-mean MAE",
                }
            ),
            "{:.3f}",
        )
        if not dflt.empty
        else "*(not available)*"
    )
    ta = pd.DataFrame(A["by_temperature_range"]).rename(columns={"temp_range": "range [°C]"})
    tb = pd.DataFrame(B["by_temperature_range"]).rename(columns={"temp_range": "range [°C]"})
    err = pd.DataFrame(
        {
            "measured range [°C]": ta["range [°C]"],
            "samples": ta["n"].map(lambda v: f"{v:,}"),
            "Track A MAE": ta["mae"],
            "Track B MAE": tb["mae"],
            "Track B p95": tb["p95_ae"],
            "Track B bias": tb["bias"],
        }
    )
    val["table_err_temp"] = md_table(err)
    reg = pd.DataFrame(B["by_operating_region"])
    piv = reg.pivot(index="speed_region", columns="torque_region", values="mae").reindex(
        index=["low", "mid", "high"], columns=["low", "mid", "high"]
    )
    piv.index = [f"{i} speed" for i in piv.index]
    piv.columns = [f"{c} torque" for c in piv.columns]
    edges = B["region_edges"]
    val["table_region_B"] = md_table(piv.reset_index().rename(columns={"index": "MAE [°C]"})) + (
        f"\n\n*Tercile edges (training data): |speed| {edges['abs_speed_rpm'][1]:.0f} / {edges['abs_speed_rpm'][2]:.0f} rpm, "
        f"|torque| {edges['abs_torque_nm'][1]:.1f} / {edges['abs_torque_nm'][2]:.1f} N·m.*"
    )
    el = pd.DataFrame(B["by_elapsed_time_in_profile"])
    val["table_elapsed_B"] = md_table(
        el.rename(columns={"elapsed": "time in profile", "n": "samples", "mae": "MAE [°C]"})
    )
    first10 = (
        float(el.loc[el.elapsed == "0-10 min", "mae"].iloc[0])
        if (el.elapsed == "0-10 min").any()
        else float("nan")
    )
    late = (
        float(el.loc[el.elapsed == ">2 h", "mae"].iloc[0])
        if (el.elapsed == ">2 h").any()
        else float("nan")
    )
    val.update(err_first10=_f(first10), err_late=_f(late))
    # ---- case study
    pp = pd.DataFrame(B["per_profile"]).set_index("profile_id")
    pid = int(r["case_studies"]["worst_profile_B"])
    story, early, late = _worst_story(pred_path, pid, g["profiles"])
    if early > 2 * late:
        story += (
            " The large early error is the cold-start phase: the 60 s / 300 s rolling windows are NaN at "
            "first and the estimate can jump when they become available (see the figure)."
        )
    val["stream_us"] = _f(_stream_us(mode_dir(mode) / "models" / "track_b.json", cfg), 0)
    val.update(
        worst_pid=pid, worst_mae=_f(pp.loc[pid, "mae"]), case_study_text=story, worst_why=story
    )
    # ---- restart robustness
    w = r["warm_start_robustness_track_B"]
    if w.get("profiles_used"):
        val["restart_text"] = (
            f"On {len(w['profiles_used'])} long test profiles the history buffer was emptied after "
            f"{w['restart_after_s'] / 60:.0f} min (motor already warm) and rebuilt from that moment. MAE over the rest of "
            f"those profiles: {w['mae_full_history']:.2f} °C with full history vs {w['mae_restarted_history']:.2f} °C after the "
            f"restart; in the first 30 min after the restart {w['mae_first_30min_after_restart_full_history']:.2f} °C vs "
            f"{w['mae_first_30min_after_restart_restarted']:.2f} °C."
        )
        val["restart_short"] = (
            f"Emptying the buffer on a warm motor raised the MAE in the next 30 min from "
            f"{w['mae_first_30min_after_restart_full_history']:.2f} to {w['mae_first_30min_after_restart_restarted']:.2f} °C."
        )
        val.update(
            restart_before=_f(w["mae_first_30min_after_restart_full_history"]),
            restart_after=_f(w["mae_first_30min_after_restart_restarted"]),
        )
    else:
        val["restart_text"] = val["restart_short"] = (
            "No test profile was long enough for the restart experiment."
        )
        val.update(restart_before="n/a", restart_after="n/a")

    # ---- cost
    def row(c: str) -> pd.Series:
        return cand[cand.candidate == c].iloc[0]

    ra, rb = row(sel["A"]), row(sel["B"])
    val.update(
        A_fit=_f(ra.fit_seconds, 1),
        A_size=f"{ra.model_size_kb:,.0f}",
        A_latency=_f(ra.single_row_latency_ms, 1),
        A_thr=f"{ra.batch_rows_per_s:,.0f}",
        B_fit=_f(rb.fit_seconds, 1),
        B_size=f"{rb.model_size_kb:,.0f}",
        B_latency=_f(rb.single_row_latency_ms, 1),
        B_thr=f"{rb.batch_rows_per_s:,.0f}",
        B_size_kb=f"{rb.model_size_kb:,.0f}",
    )
    # ---- explainability
    gb = g["impg_b"].rename(
        columns={"importance_mae_increase": "MAE increase [°C]", "std": "± std"}
    )
    val["table_impg_B"] = md_table(gb[["group", "n_features", "MAE increase [°C]", "± std"]])
    ga = g["impg_a"].rename(
        columns={"importance_mae_increase": "MAE increase [°C]", "std": "± std"}
    )
    val["table_impg_A"] = md_table(ga[["group", "n_features", "MAE increase [°C]", "± std"]])
    ib = (
        g["imp_b"]
        .head(10)
        .rename(columns={"importance_mae_increase": "MAE increase [°C]", "std": "± std"})
    )
    val["table_imp_B"] = md_table(ib[["feature", "group", "MAE increase [°C]", "± std"]])
    top = g["impg_b"].iloc[:2]
    val["importance_text"] = (
        f"For Track B the largest increases come from the families '{top.iloc[0].group}' "
        f"(+{top.iloc[0].importance_mae_increase:.1f} °C) and '{top.iloc[1].group}' "
        f"(+{top.iloc[1].importance_mae_increase:.1f} °C), whereas the present-moment families add comparatively little - "
        "the history features carry the information."
    )
    val["importance_short"] = (
        f"Track B relies mostly on '{top.iloc[0].group}' and '{top.iloc[1].group}'; Track A on "
        f"'{g['impg_a'].iloc[0].group}'."
    )
    # ---- monitoring
    mv, mt = mon["validation"], mon["test"]
    val.update(
        smooth_s=mon["smoothing_window_s"],
        band_q=_f(mon["band_quantile"] * 100, 1),
        band=_f(mt["band_smoothed_c"], 1),
        persist_s=mon["persistence_s"],
        val_raw=_pct(mv["raw_exceed_frac"]),
        test_raw=_pct(mt["raw_exceed_frac"]),
        val_smooth=_pct(mv["smooth_exceed_frac"]),
        test_smooth=_pct(mt["smooth_exceed_frac"]),
        val_alarm=_pct(mv["persistent_alarm_frac"]),
        test_alarm=_pct(mt["persistent_alarm_frac"]),
        val_prof_alarm=mv["profiles_with_any_alarm"],
        test_prof_alarm=mt["profiles_with_any_alarm"],
        val_events=_f(mv["alarm_events_per_hour"], 2),
        test_events=_f(mt["alarm_events_per_hour"], 2),
    )
    ratio = mt["persistent_alarm_frac"] / max(mv["persistent_alarm_frac"], 1e-9)
    val["monitor_transfer_text"] = (
        f"the persistent-alarm time is {_pct(mv['persistent_alarm_frac'])} % on validation and "
        f"{_pct(mt['persistent_alarm_frac'])} % on test ({ratio:.0f}x the validation value, though small in absolute "
        f"terms), and {mt['profiles_with_any_alarm']} of {mt['n_profiles']} test profiles raise at least one alarm."
    )
    inj = pd.DataFrame(mon["synthetic_injected_offset_on_test"])
    inj_t = pd.DataFrame(
        {
            "injected offset [°C]": inj["offset_c"],
            "profiles alarming": [
                f"{a} / {n}" for a, n in zip(inj.profiles_alarming, inj.n_profiles, strict=True)
            ],
            "of which newly (not alarming without offset)": inj["profiles_newly_alarming"],
            "median first alarm [min]": [
                ("-" if pd.isna(v) else f"{v / 60:.0f}") for v in inj.median_first_alarm_s
            ],
            "time in alarm [%]": [f"{v * 100:.1f}" for v in inj.alarm_frac_of_samples],
        }
    )
    val["table_injection"] = md_table(inj_t)
    val["inj_sentence"] = sent["inj_sentence"]
    # ---- forecast
    fc = g["forecast"]
    val["horizon"] = cfg["forecast"]["horizon_s"]
    if fc:
        rows = []
        for key, label in (
            ("persistence_T_now", "persistence T(t)"),
            ("track_B_estimate_used_as_forecast", "Track B estimate used as forecast (wrong tool)"),
            ("F1_sensor_independent", "F1 - forecast model without winding sensor"),
            ("F2_sensor_assisted", "F2 - forecast model with measured T(t) and its change"),
        ):
            a = fc[key]["aggregate"]
            rows.append(
                {
                    "method": label,
                    "MAE [°C]": a["mae"],
                    "RMSE [°C]": a["rmse"],
                    "R²": a["r2"],
                    "worst-profile MAE": fc[key]["profile_level"]["worst_profile_mae"],
                }
            )
        val["table_forecast"] = md_table(pd.DataFrame(rows), "{:.3f}")
        agg = {
            k: fc[k]["aggregate"]
            for k in ("persistence_T_now", "F1_sensor_independent", "F2_sensor_assisted")
        }
        pe, f1, f2 = (
            agg[k]["mae"]
            for k in ("persistence_T_now", "F1_sensor_independent", "F2_sensor_assisted")
        )
        better_rmse = agg["F1_sensor_independent"]["rmse"] < agg["persistence_T_now"]["rmse"]
        f1_text = (
            f"The sensor-independent forecast F1 ({f1:.2f} °C) {'beats' if f1 < pe else 'does not beat'} persistence "
            f"({pe:.2f} °C) on MAE; its RMSE is {agg['F1_sensor_independent']['rmse']:.2f} °C vs "
            f"{agg['persistence_T_now']['rmse']:.2f} °C for persistence"
            + (" (better on large errors)" if better_rmse else "")
            + ". "
        )
        f2_text = (
            f"The sensor-assisted forecast F2 ({f2:.2f} °C) {'beats' if f2 < pe else 'does not beat'} persistence, "
            "which shows how valuable the present measured state is. "
        )
        val["forecast_text"] = (
            f"Target: the winding temperature {cfg['forecast']['horizon_s']} s later (shifted within each profile; the last "
            f"{cfg['forecast']['horizon_s']} s of every profile have no label and are dropped), causal features only - no future "
            "current, speed, torque, coolant or temperature. F1 uses the Track B features; F2 additionally uses the "
            "*measured* winding temperature at time t and its change over the last horizon (needs the physical sensor). "
            f"Persistence MAE: {pe:.2f} °C. "
            + f1_text
            + f2_text
            + "The Track B estimator, which targets the *current* temperature, is not a forecaster "
            f"(MAE {fc['track_B_estimate_used_as_forecast']['aggregate']['mae']:.2f} °C against the future value). "
            "This extension is a small, separate experiment on the same held-out test profiles, with no tuning on them."
        )
        val.update(
            fc_persist=_f(pe),
            fc_f1=_f(f1),
            fc_f2=_f(f2),
            fc_estimator=_f(fc["track_B_estimate_used_as_forecast"]["aggregate"]["mae"]),
        )
    else:
        val["table_forecast"] = ""
        val["forecast_text"] = "The optional forecasting experiment was not run."
        for k in ("fc_persist", "fc_f1", "fc_f2", "fc_estimator"):
            val[k] = "n/a"
    # ---- ablation numbers for viva
    none = ab[ab.history_set == "none"].iloc[0]
    best = ab[ab.history_set != "none"].sort_values("val_mae_seed_mean").iloc[0]
    val.update(
        abl_none=_f(none.val_mae_seed_mean),
        abl_best=_f(best.val_mae_seed_mean),
        abl_std=_f(best.val_mae_seed_std),
    )
    val["conclusion_sentence"] = sent["conclusion_sentence"]
    return val


def write_documents(
    cfg: dict[str, Any], mode: str, g: dict[str, Any], sent: dict[str, str]
) -> dict[str, Path]:
    values = build_values(cfg, mode, g, sent)
    out: dict[str, Path] = {}
    report = render((TEMPLATES / "project_report.md.tmpl").read_text(encoding="utf-8"), values)
    out["report"] = REPORTS_DIR / "project_report.md"
    out["report"].write_text(report, encoding="utf-8")
    viva = render((TEMPLATES / "viva.md.tmpl").read_text(encoding="utf-8"), values)
    out["viva"] = PROJECT_ROOT / "docs" / "VIVA_PREPARATION.md"
    out["viva"].write_text(viva, encoding="utf-8")
    # README results block
    readme = PROJECT_ROOT / "README.md"
    block = (
        "<!-- RESULTS:START -->\n"
        f"**Results ({values['mode_label']}; {values['n_test']} unseen test profiles, {g['res']['test_rows']:,} samples)** - "
        "generated by `motor-thermal report`:\n\n"
        "| | Track A: instantaneous | Track B: history-aware |\n|---|---|---|\n"
        f"| MAE [°C] | {values['A_mae']} | **{values['B_mae']}** |\n"
        f"| RMSE [°C] | {values['A_rmse']} | {values['B_rmse']} |\n"
        f"| R² | {values['A_r2']} | {values['B_r2']} |\n"
        f"| 95th-pct abs. error [°C] | {values['A_p95']} | {values['B_p95']} |\n"
        f"| worst test profile MAE [°C] | {values['A_worst']} | {values['B_worst']} |\n"
        f"| winner | `{values['win_A']}` | `{values['win_B']}` |\n\n"
        "See `reports/project_report.md` for the full evaluation, ablation, explainability and monitoring.\n"
        "<!-- RESULTS:END -->"
    )
    text = readme.read_text(encoding="utf-8")
    text = re.sub(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _m: block, text, flags=re.S
    )
    readme.write_text(text, encoding="utf-8")
    out["readme"] = readme
    return out


__all__ = ["ARTIFACTS_DIR", "write_documents"]
