"""Exploratory data analysis. Statistics that could steer modelling use TRAIN profiles only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import data as data_mod  # noqa: E402
from . import plots  # noqa: E402
from . import splits as spl  # noqa: E402
from .config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR  # noqa: E402
from .evaluate import dump_json  # noqa: E402

log = logging.getLogger(__name__)

TEMPS = ["stator_winding", "stator_tooth", "stator_yoke", "pm", "coolant", "ambient"]
CORR_COLS = [
    "ambient",
    "coolant",
    "motor_speed",
    "torque",
    "i_d",
    "i_q",
    "u_d",
    "u_q",
    "i_sq",
    "stator_winding",
    "stator_tooth",
    "stator_yoke",
    "pm",
]


def constant_runs(values: np.ndarray, min_len: int) -> tuple[int, int]:
    """(# runs, # samples) of >= ``min_len`` exactly identical consecutive values."""
    same = np.r_[False, values[1:] == values[:-1]]
    runs = samples = 0
    k = 0
    for s in np.r_[same, False]:
        if s:
            k += 1
        else:
            if k + 1 >= min_len:
                runs += 1
                samples += k + 1
            k = 0
    return runs, samples


def run_eda(cfg: dict[str, Any], csv: str | Path | None = None) -> dict[str, Any]:
    cols = cfg["columns"]
    group, target = cols["group"], cols["target"]
    hz = float(cfg["dataset"]["sampling_hz"])
    path = data_mod.locate_dataset(cfg, csv)
    df = data_mod.load_measurements(path, cfg)
    rep = data_mod.validate_schema(df, cfg)
    profiles = data_mod.profile_table(df, cfg)
    manifest = spl.make_split_manifest(profiles, cfg)
    split_of = {p: s for s in spl.SPLITS for p in manifest[s]["profiles"]}
    profiles["split"] = profiles.index.map(split_of)
    df["i_sq"] = (df["i_d"].astype("float64") ** 2 + df["i_q"].astype("float64") ** 2).astype(
        "float32"
    )
    train = spl.split_frame(df, manifest, "train", group)
    fig = FIGURES_DIR / "eda"
    REPORTS_DIR.mkdir(exist_ok=True)

    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "profiles": int(df[group].nunique()),
        "hours_at_assumed_2hz": round(len(df) / hz / 3600, 2),
        "validation": {"ok": rep.ok, "warnings": rep.warnings, "errors": rep.errors, **rep.facts},
        "profile_length_rows": profiles["rows"].describe().round(1).to_dict(),
        "profile_duration_h": profiles["duration_h"].describe().round(3).to_dict(),
        "describe": df[[c for c in df.columns if c != "i_sq"]]
        .describe()
        .T.round(3)
        .to_dict("index"),
    }

    # --- chronology / sampling sanity (no timestamps shipped)
    step = df.groupby(group)[target].diff().abs()
    summary["sampling_sanity"] = {
        "winding_abs_step_c_median": float(step.median()),
        "winding_abs_step_c_p99.9": float(step.quantile(0.999)),
        "winding_abs_step_c_max": float(step.max()),
        "interpretation": "Smooth per-sample steps are consistent with a 2 Hz series; the file has no "
        "timestamps, so the rate is taken from the dataset documentation.",
    }

    # --- constant readings / duplicates (inspected, not blindly removed)
    runs = [constant_runs(g[target].to_numpy(), 20) for _, g in df.groupby(group)]
    speed0 = [(g["motor_speed"].abs() < 1).mean() for _, g in df.groupby(group)]
    summary["constant_and_duplicate_readings"] = {
        "duplicate_rows": rep.facts["duplicate_rows"],
        "winding_constant_runs_ge_10s": int(sum(r[0] for r in runs)),
        "winding_samples_in_constant_runs": int(sum(r[1] for r in runs)),
        "fraction_of_rows_with_|speed|<1rpm": float((df["motor_speed"].abs() < 1).mean()),
        "profiles_with_>50%_standstill": int(sum(s > 0.5 for s in speed0)),
        "treatment": "kept: standstill/idle periods are legitimate operating conditions",
    }

    # --- start state (cold vs warm start)
    start_gap = profiles["winding_start"] - df.groupby(group)["ambient"].first()
    summary["start_state"] = {
        "profiles_starting_within_5C_of_ambient": int((start_gap < 5).sum()),
        "profiles_starting_warm_(>5C_above_ambient)": int((start_gap >= 5).sum()),
        "max_start_winding_c": float(profiles["winding_start"].max()),
    }

    # --- physical plausibility
    summary["plausibility"] = {
        c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in cols["inputs"] + [target]
    }

    corr = train[CORR_COLS].corr()
    corr.round(3).to_csv(ARTIFACTS_DIR / "eda_correlation_train.csv")
    summary["corr_with_winding_train"] = corr[target].drop(target).round(3).to_dict()
    summary["corr_torque_iq_train"] = round(float(corr.loc["torque", "i_q"]), 4)

    # ------------------------------------------------------------ figures
    # 1 profile durations coloured by split
    fig1, ax = plt.subplots(figsize=(11, 3.8))
    order = profiles.sort_index()
    colors = {"train": plots.C_B, "val": plots.C_A, "test": plots.C_BAD}
    ax.bar(order.index.astype(str), order["duration_h"], color=[colors[s] for s in order["split"]])
    handles = [
        Patch(color=c, label=f"{s} ({manifest[s]['n_profiles']} profiles)")
        for s, c in colors.items()
    ]
    ax.set(
        xlabel="profile_id",
        ylabel="duration [h] at 2 Hz",
        title="Operating-profile durations and split assignment",
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(handles=handles)
    plots._save(fig1, fig / "profile_durations.png")

    # 2 temperature distributions
    fig2, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), TEMPS, strict=True):
        ax.hist(train[c], bins=100, color=plots.C_B if c == target else plots.C_GREY, alpha=0.9)
        ax.set(title=c, xlabel="°C", ylabel="samples")
    fig2.suptitle("Temperature distributions (training profiles)", fontweight="bold")
    plots._save(fig2, fig / "temperature_distributions.png")

    # 3 correlation heatmap
    fig3, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
    ax.grid(False)
    fig3.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation (training profiles)")
    plots._save(fig3, fig / "correlation_heatmap.png")

    # 4 operating-region coverage by split
    fig4, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for s in ("train", "val", "test"):
        sub = spl.split_frame(df, manifest, s, group).iloc[::40]
        axes[0].scatter(
            sub["motor_speed"], sub["torque"], s=2, alpha=0.25, color=colors[s], label=s
        )
        axes[1].scatter(sub["i_d"], sub["i_q"], s=2, alpha=0.25, color=colors[s], label=s)
    axes[0].set(xlabel="motor speed [rpm]", ylabel="torque [Nm]", title="Speed-torque coverage")
    axes[1].set(xlabel="i_d [A]", ylabel="i_q [A]", title="d-q current plane coverage")
    axes[0].legend(markerscale=6)
    plots._save(fig4, fig / "operating_coverage.png")

    # 5 example time series: longest train profile, a warm-start train profile
    tp = profiles[profiles["split"] == "train"]
    examples = {
        "cold start": int(
            tp[tp["winding_start"] < tp["winding_start"].quantile(0.3)]["rows"].idxmax()
        ),
        "warm start": int(tp["winding_start"].idxmax()),
    }
    fig5, axes = plt.subplots(len(examples), 1, figsize=(11, 7), sharex=False)
    for ax, (tag, pid) in zip(axes, examples.items(), strict=True):
        g = df[df[group] == pid]
        t = np.arange(len(g)) / hz / 60
        ax.plot(t, g[target], color=plots.C_MEAS, label="stator_winding (target)")
        ax.plot(t, g["coolant"], color=plots.C_B, lw=0.9, label="coolant")
        ax2 = ax.twinx()
        ax2.plot(t, g["motor_speed"], color=plots.C_A, lw=0.4, alpha=0.5, label="speed [rpm]")
        ax2.grid(False)
        ax.set(title=f"profile {pid} ({tag})", ylabel="°C", xlabel="time [min]")
        ax2.set_ylabel("speed [rpm]")
        ax.legend(loc="upper left", fontsize=8)
    plots._save(fig5, fig / "example_timeseries.png")

    # 6 winding vs coolant and current relationship
    fig6, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sub = train.iloc[::30]
    axes[0].hexbin(sub["coolant"], sub[target], gridsize=60, bins="log", cmap="Blues", mincnt=1)
    axes[0].set(
        xlabel="coolant [°C]", ylabel="winding [°C]", title="Winding vs coolant temperature"
    )
    axes[1].hexbin(sub["i_sq"], sub[target], gridsize=60, bins="log", cmap="Blues", mincnt=1)
    axes[1].set(
        xlabel="i_d² + i_q² [A²]",
        ylabel="winding [°C]",
        title="Winding vs squared current (instantaneous)",
    )
    axes[2].hexbin(
        sub["stator_winding"], sub["stator_tooth"], gridsize=60, bins="log", cmap="Blues", mincnt=1
    )
    axes[2].set(
        xlabel="winding [°C]",
        ylabel="tooth [°C]",
        title="Internal temperatures are near-collinear\n(why they are excluded as inputs)",
    )
    plots._save(fig6, fig / "winding_relationships.png")

    # 7 start temperature
    fig7, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(profiles["winding_start"], bins=25, color=plots.C_B)
    ax.set(
        xlabel="winding temperature at first sample [°C]",
        ylabel="profiles",
        title="Profiles do not all start cold",
    )
    plots._save(fig7, fig / "start_temperature.png")

    profiles.to_csv(ARTIFACTS_DIR / "profile_table.csv")
    dump_json(summary, REPORTS_DIR / "eda_summary.json")
    log.info("EDA written to %s", fig)
    return summary
