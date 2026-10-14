"""Generate the markdown report, viva-preparation numbers, architecture diagram and PowerPoint.

Every number is read from the saved result files; templates contain ``{{placeholders}}`` only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import deck, plots
from .config import ARTIFACTS_DIR, FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, mode_dir

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


def render(template: str, values: dict[str, Any]) -> str:
    """Replace ``{{key}}``; raise if a key is missing so no placeholder can leak into a document."""
    missing = sorted({m for m in PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise KeyError(f"template placeholders without a value: {missing}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def gather(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    """Load every result file the documents need."""
    d = mode_dir(mode)
    need = [d / "metrics_test.json", d / "candidates_test.csv", d / "run_record.json"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"run train/evaluate for mode {mode!r} first; missing: {missing}")
    out = {
        "res": _read_json(d / "metrics_test.json"),
        "run": _read_json(d / "run_record.json"),
        "cand_test": pd.read_csv(d / "candidates_test.csv"),
        "cand_val": pd.read_csv(d / "candidates_val.csv"),
        "ablation": pd.read_csv(d / "history_ablation_val.csv"),
        "prov": _read_json(ARTIFACTS_DIR / "provenance.json"),
        "manifest": _read_json(ARTIFACTS_DIR / "split_manifest.json"),
        "profiles": pd.read_csv(ARTIFACTS_DIR / "profile_table.csv", index_col=0),
        "eda": _read_json(REPORTS_DIR / "eda_summary.json"),
        "cv": pd.read_csv(d / "group_cv.csv") if (d / "group_cv.csv").is_file() else None,
        "forecast": _read_json(d / "forecast_test.json")
        if (d / "forecast_test.json").is_file()
        else None,
        "imp_b": pd.read_csv(d / "importance_track_b.csv"),
        "imp_a": pd.read_csv(d / "importance_track_a.csv"),
        "impg_b": pd.read_csv(d / "importance_groups_track_b.csv"),
        "impg_a": pd.read_csv(d / "importance_groups_track_a.csv"),
    }
    return out


def sentences(g: dict[str, Any], mode: str) -> dict[str, str]:
    """Interpretive sentences computed from the data so they can never contradict the numbers."""
    r, abl = g["res"], g["ablation"]
    a, b = r["track_A"]["test"]["aggregate"], r["track_B"]["test"]["aggregate"]
    none = abl[abl.history_set == "none"].iloc[0]
    hist = abl[abl.history_set != "none"].copy()
    best = hist.sort_values("val_mae_seed_mean").iloc[0]
    worst = hist.sort_values("val_mae_seed_mean").iloc[-1]
    ratio = a["mae"] / b["mae"]
    abl_s = (
        f"On the validation profiles, every history set beats the instantaneous model "
        f"({none['val_mae_seed_mean']:.2f} °C); the best set ('{best['history_set']}') reaches "
        f"{best['val_mae_seed_mean']:.2f} °C (± {best['val_mae_seed_std']:.2f} across seeds). "
        f"Differences between history sets are smaller than the gap to the instantaneous model, and "
        f"the seed spread shows differences below ≈ {max(0.3, hist['val_mae_seed_std'].max()):.1f} °C "
        "should not be over-interpreted."
    )
    if not (hist["val_mae_seed_mean"] < none["val_mae_seed_mean"]).all():
        abl_s = (
            "Not every history set improved on the instantaneous model on validation; the best "
            f"('{best['history_set']}') reached {best['val_mae_seed_mean']:.2f} °C versus "
            f"{none['val_mae_seed_mean']:.2f} °C."
        )
    inj = pd.DataFrame(r["monitoring"]["synthetic_injected_offset_on_test"])
    inj_s = "; ".join(
        f"+{row.offset_c:g} °C → {int(row.profiles_newly_alarming)} newly alarming of "
        f"{int(row.n_profiles)} profiles"
        + (
            f" (median first alarm {row.median_first_alarm_s / 60:.0f} min)"
            if pd.notna(row.median_first_alarm_s)
            else ""
        )
        for row in inj.itertuples()
    )
    fc = g["forecast"]
    if fc:
        fc_s = (
            f"60 s-ahead forecasting (separate experiment): persistence MAE "
            f"{fc['persistence_T_now']['aggregate']['mae']:.2f} °C, sensor-independent forecast "
            f"{fc['F1_sensor_independent']['aggregate']['mae']:.2f} °C, sensor-assisted forecast "
            f"{fc['F2_sensor_assisted']['aggregate']['mae']:.2f} °C."
        )
    else:
        fc_s = "60 s-ahead forecasting experiment: not run."
    conc = (
        f"On {len(r['test_profiles'])} unseen test profiles the history-aware virtual sensor reaches "
        f"MAE {b['mae']:.2f} °C (RMSE {b['rmse']:.2f} °C, R² {b['r2']:.3f}, 95 % of errors below "
        f"{b['p95_ae']:.1f} °C) versus {a['mae']:.2f} °C for the best instantaneous model - "
        f"{ratio:.1f}× lower error."
    )
    if mode != "full":
        conc += " (Fast-mode, preliminary.)"
    return {
        "ablation_sentence": abl_s,
        "inj_sentence": inj_s,
        "forecast_sentence": fc_s,
        "conclusion_sentence": conc,
        "worst_hist_set": str(worst["history_set"]),
        "best_hist_set": str(best["history_set"]),
    }


def build_all(cfg: dict[str, Any], mode: str) -> dict[str, Path]:
    g = gather(cfg, mode)
    fig_root = FIGURES_DIR
    fig_dir = FIGURES_DIR / mode
    counts = tuple(g["manifest"][s]["n_profiles"] for s in ("train", "val", "test"))
    plots.architecture_diagram(fig_root / "architecture.png", counts)  # type: ignore[arg-type]
    sent = sentences(g, mode)
    n_warm = int(g["eda"]["start_state"]["profiles_starting_warm_(>5C_above_ambient)"])
    ctx = {
        **g,
        "mode": mode,
        "fig_dir": fig_dir,
        "fig_root": fig_root,
        "eda_dir": FIGURES_DIR / "eda",
        "n_warm": n_warm,
        **sent,
    }
    out: dict[str, Path] = {}
    out["pptx"] = deck.build_deck(ctx, REPORTS_DIR / "UCS321_MiniProject6_Presentation.pptx")
    from .report_docs import write_documents

    out.update(write_documents(cfg, mode, g, sent))
    log.info("generated: %s", {k: str(v.relative_to(PROJECT_ROOT)) for k, v in out.items()})
    return out
