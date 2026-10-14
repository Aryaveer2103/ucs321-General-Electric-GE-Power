"""Matplotlib figures used by the report, presentation and documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C_MEAS = "#111827"
C_A = "#d97706"
C_B = "#2563eb"
C_GREY = "#6b7280"
C_BAD = "#dc2626"
C_OK = "#059669"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )


_style()


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pred_vs_actual(y: np.ndarray, p: np.ndarray, title: str, path: Path, color: str = C_B) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    hb = ax.hexbin(y, p, gridsize=70, bins="log", cmap="Blues", mincnt=1)
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], "--", color=C_BAD, lw=1.2, label="ideal (y = x)")
    ax.set(xlabel="Measured winding temperature [°C]", ylabel="Estimated [°C]", title=title)
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="samples (log)")
    return _save(fig, path)


def timeseries(
    t_min: np.ndarray,
    measured: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    path: Path,
    extra: dict[str, np.ndarray] | None = None,
) -> Path:
    n = 2 if extra else 1
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.4 * n), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(t_min, measured, color=C_MEAS, lw=1.6, label="measured (reference)")
    colors = {"A": C_A, "B": C_B}
    for name, v in preds.items():
        ax.plot(t_min, v, lw=1.1, color=colors.get(name[:1], C_GREY), label=name, alpha=0.9)
    ax.set(ylabel="Winding temperature [°C]", title=title)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    if extra:
        ax2 = axes[1, 0]
        for name, v in extra.items():
            ax2.plot(t_min, v, lw=0.9, label=name)
        ax2.set(xlabel="time in profile [min]", ylabel="driving signals")
        ax2.legend(ncol=4, fontsize=8, loc="upper left")
    else:
        ax.set_xlabel("time in profile [min]")
    return _save(fig, path)


def residual_diagnostics(y: np.ndarray, p: np.ndarray, path: Path, label: str) -> Path:
    res = y - p
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(res, bins=120, color=C_B, alpha=0.85)
    axes[0].axvline(0, color=C_MEAS, lw=1)
    axes[0].set(
        xlabel="residual = measured − estimated [°C]",
        ylabel="samples",
        title=f"Residual distribution ({label})",
    )
    hb = axes[1].hexbin(p, res, gridsize=70, bins="log", cmap="Blues", mincnt=1)
    axes[1].axhline(0, color=C_BAD, lw=1)
    axes[1].set(
        xlabel="estimated temperature [°C]", ylabel="residual [°C]", title="Residual vs estimate"
    )
    fig.colorbar(hb, ax=axes[1], label="samples (log)")
    return _save(fig, path)


def per_profile_bars(pp: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    names = list(pp)
    ids = sorted(set().union(*[set(d.index) for d in pp.values()]))
    x = np.arange(len(ids))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"A": C_A, "B": C_B}
    for i, n in enumerate(names):
        ax.bar(
            x + i * w,
            [pp[n]["mae"].get(p, np.nan) for p in ids],
            w,
            label=n,
            color=colors.get(n[:1], C_GREY),
        )
    ax.set_xticks(x + w * (len(names) - 1) / 2, [str(p) for p in ids])
    ax.set(xlabel="held-out profile_id", ylabel="MAE [°C]", title=title)
    ax.legend()
    return _save(fig, path)


def importance_bar(df: pd.DataFrame, path: Path, title: str, top: int = 15) -> Path:
    d = df.sort_values("importance_mae_increase", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.4))
    ax.barh(d["feature"], d["importance_mae_increase"], xerr=d.get("std"), color=C_B, alpha=0.9)
    ax.set(xlabel="MAE increase when the feature is shuffled [°C]", title=title)
    return _save(fig, path)


def ablation_bar(df: pd.DataFrame, path: Path) -> Path:
    """Validation MAE per history set; mean +- std over random seeds when available."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    mean_col = "val_mae_seed_mean" if "val_mae_seed_mean" in df else "val_mae"
    labels = df["history_set"].str.replace("_", "\n", n=1)
    cols = [C_A if h == "none" else C_B for h in df["history_set"]]
    err = df["val_mae_seed_std"] if "val_mae_seed_std" in df else None
    bars = ax.bar(labels, df[mean_col], yerr=err, capsize=4, color=cols)
    for b, n in zip(bars, df["n_features"], strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.15,
            f"{b.get_height():.2f}\n({n} feat.)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df else 1
    ax.set(
        ylabel="validation MAE [°C]",
        title=f"History ablation - LightGBM, same validation profiles (mean ± std, {seeds} seeds)",
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, (df[mean_col].max() + (err.max() if err is not None else 0)) * 1.3)
    return _save(fig, path)


def model_comparison(
    table: pd.DataFrame, path: Path, col: str = "test_mae", title: str = ""
) -> Path:
    d = table.sort_values(col, ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(d) + 1.4))
    colors = [C_B if t == "B" else (C_A if t == "A" else C_GREY) for t in d["track"]]
    ax.barh(d["candidate"], d[col], color=colors)
    for i, v in enumerate(d[col]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set(xlabel="MAE [°C]", title=title or "Model comparison")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


def error_by_range(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(tables)
    bins = tables[names[0]].iloc[:, 0].tolist()
    x = np.arange(len(bins))
    w = 0.8 / len(names)
    colors = {"A": C_A, "B": C_B}
    for i, n in enumerate(names):
        ax.bar(x + i * w, tables[n]["mae"], w, label=n, color=colors.get(n[:1], C_GREY))
    ax.set_xticks(x + w * (len(names) - 1) / 2, bins)
    ax.set(
        xlabel="measured winding temperature range [°C]",
        ylabel="MAE [°C]",
        title="Error by temperature range (test)",
    )
    ax.legend()
    return _save(fig, path)


def monitoring_plot(
    t_min: np.ndarray,
    measured: np.ndarray,
    est: np.ndarray,
    smooth_res: np.ndarray,
    band: float,
    flags: np.ndarray,
    path: Path,
    title: str,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_min, measured, color=C_MEAS, lw=1.5, label="measured")
    axes[0].plot(t_min, est, color=C_B, lw=1.1, label="estimated (virtual sensor)")
    axes[0].set(ylabel="Winding temperature [°C]", title=title)
    axes[0].legend(loc="upper left")
    axes[1].plot(t_min, smooth_res, color=C_MEAS, lw=1)
    axes[1].axhspan(-band, band, color=C_OK, alpha=0.12, label=f"band ±{band:.1f} °C (validation)")
    if flags.any():
        axes[1].fill_between(
            t_min,
            -band * 3,
            band * 3,
            where=flags,
            color=C_BAD,
            alpha=0.15,
            label="persistent alarm",
        )
    lim = max(np.nanmax(np.abs(smooth_res)) * 1.15, band * 1.5)
    axes[1].set(ylim=(-lim, lim), xlabel="time in profile [min]", ylabel="smoothed residual [°C]")
    axes[1].legend(loc="upper left", fontsize=8)
    return _save(fig, path)


def architecture_diagram(path: Path, counts: tuple[int, int, int] = (49, 10, 10)) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#374151", fs=9, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.2
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            wrap=True,
        )

    def arrow(x1, y1, x2, y2, color="#374151"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4
            )
        )

    ax.text(
        0.1, 5.95, "Virtual winding-temperature sensor - pipeline", fontsize=13, fontweight="bold"
    )
    # row 1: data
    box(
        0.1,
        4.3,
        2.3,
        1.2,
        "Paderborn LEA\nmeasures_v2.csv\n1.33 M rows, 69 profiles\n(kagglehub / manual CSV)",
        "#dbeafe",
    )
    box(
        2.9, 4.3, 2.3, 1.2, "Validation &\nprovenance\nschema, NaN, contiguity,\nSHA-256", "#e0e7ff"
    )
    box(
        5.7,
        4.3,
        2.3,
        1.2,
        f"Profile-wise split\n{counts[0]} / {counts[1]} / {counts[2]} profiles\n"
        "(stratified by peak temp.,\nmanifest saved)",
        "#fef3c7",
    )
    box(8.5, 4.3, 2.0, 1.2, "EDA\ndistributions, correlations,\ncoverage, time series", "#e0e7ff")
    arrow(2.4, 4.9, 2.9, 4.9)
    arrow(5.2, 4.9, 5.7, 4.9)
    arrow(8.0, 4.9, 8.5, 4.9)
    # row 2: features
    box(
        0.1,
        2.5,
        3.0,
        1.2,
        "Instantaneous features\ni², |u|, |ω|, |T|, P_mech,\nP_loss proxy, T_amb − T_cool",
        "#dcfce7",
    )
    box(
        3.6,
        2.5,
        3.6,
        1.2,
        "Causal history features (per profile)\nEWMA τ = 30 s … 2 h, rolling means,\nof i², |ω|, |T|, coolant",
        "#dcfce7",
    )
    box(
        7.7,
        2.5,
        2.8,
        1.2,
        "Forbidden: winding / tooth /\nyoke / magnet temperature,\nprofile_id as feature",
        "#fee2e2",
        ec=C_BAD,
    )
    arrow(6.8, 4.3, 1.6, 3.7)
    arrow(3.1, 3.1, 3.6, 3.1)
    # row 3: models
    box(
        0.1,
        0.5,
        3.4,
        1.3,
        "Track A - instantaneous\nbaselines, Linear, Ridge, RF,\nHistGB, LightGBM",
        "#fde68a",
    )
    box(
        4.0,
        0.5,
        3.4,
        1.3,
        "Track B - history-aware\nLightGBM / HistGB on\ninstantaneous + causal history",
        "#bfdbfe",
    )
    arrow(1.6, 2.5, 1.8, 1.8)
    arrow(5.4, 2.5, 5.7, 1.8)
    box(
        7.9,
        0.5,
        2.6,
        1.3,
        "Held-out test profiles\nMAE, RMSE, R², p95,\nper-profile, regions",
        "#fef3c7",
    )
    arrow(3.5, 1.15, 4.0, 1.15)
    arrow(7.4, 1.15, 7.9, 1.15)
    # right column
    box(
        10.9,
        3.3,
        2.0,
        1.6,
        "Saved artifacts\nmodels + metadata,\nmetrics JSON,\nsplit manifest",
        "#e5e7eb",
    )
    box(
        10.9,
        1.2,
        2.0,
        1.6,
        "Streamlit app\nestimate, replay,\nresidual monitor",
        "#e5e7eb",
        bold=True,
    )
    box(10.9, 0.05, 2.0, 0.9, "Report / PPTX", "#e5e7eb")
    arrow(10.5, 1.15, 10.9, 3.5)
    arrow(11.9, 3.3, 11.9, 2.8)
    arrow(11.9, 1.2, 11.9, 0.95)
    return _save(fig, path)
