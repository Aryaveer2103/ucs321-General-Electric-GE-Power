"""Streamlit dashboard: PMSM winding-temperature virtual sensor.

Run:  uv run streamlit run app/streamlit_app.py

The app only uses real trained artifacts (artifacts/<mode>/models) and, for replays, the real
Paderborn recordings found on disk. Without them it shows an explicit setup message - it never
invents predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motor_thermal import data as data_mod  # noqa: E402
from motor_thermal import features as feat  # noqa: E402
from motor_thermal import inference as inf  # noqa: E402
from motor_thermal import monitor as mon  # noqa: E402
from motor_thermal.config import ARTIFACTS_DIR, FIGURES_DIR, load_config  # noqa: E402

st.set_page_config(page_title="PMSM Thermal Intelligence", page_icon="🌡️", layout="wide")

CFG = load_config()
GROUP, TARGET = CFG["columns"]["group"], CFG["columns"]["target"]
HZ = float(CFG["dataset"]["sampling_hz"])
BLUE, ORANGE, DARK, RED, GREEN = "#2563eb", "#d97706", "#111827", "#dc2626", "#059669"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1250px;}
    .note {background:#fef3c7; border-left:4px solid #d97706; padding:.6rem .9rem; border-radius:4px;
           color:#1f2937; font-size:.92rem;}
    .est {background:#dbeafe; border-left:4px solid #2563eb; padding:.6rem .9rem; border-radius:4px;
          color:#1f2937; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading trained model…")
def get_sensor(mode: str, track: str) -> inf.VirtualSensor:
    return inf.VirtualSensor.load(mode, track)


@st.cache_data(show_spinner="Loading recorded measurements…")
def get_dataset() -> pd.DataFrame | None:
    path = data_mod.find_local_dataset(CFG)
    if path is None:
        return None
    return data_mod.load_measurements(path, CFG)


@st.cache_data
def get_manifest() -> dict | None:
    p = ARTIFACTS_DIR / "split_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


@st.cache_data(show_spinner="Computing estimates for the selected profile…")
def profile_predictions(mode: str, pid: int) -> pd.DataFrame:
    df = get_dataset()
    g = df[df[GROUP] == pid].reset_index(drop=True)
    out = pd.DataFrame(
        {"t_min": np.arange(len(g)) / HZ / 60.0, "measured": g[TARGET].to_numpy("float64")}
    )
    for track in ("A", "B"):
        out[f"pred_{track}"] = get_sensor(mode, track).predict_frame(g)
    for c in ("motor_speed", "torque", "coolant", "ambient", "i_d", "i_q"):
        out[c] = g[c].to_numpy("float64")
    return out


def split_of(pid: int, manifest: dict | None) -> str:
    if not manifest:
        return "?"
    for s in ("train", "val", "test"):
        if pid in manifest[s]["profiles"]:
            return s
    return "?"


# ------------------------------------------------------------------ sidebar
modes = inf.available_modes()
st.sidebar.title("🌡️ PMSM Thermal Intelligence")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Dataset exploration",
        "Instantaneous prediction",
        "Profile replay",
        "Model evaluation",
        "Thermal monitoring",
    ],
)
if not modes:
    st.title("PMSM winding-temperature virtual sensor")
    st.error("No trained models found - the app will not invent predictions.")
    st.markdown(
        "Train them first (downloads the public dataset, ~120 MB, if needed):\n\n"
        "```bash\nuv run motor-thermal train --mode fast\nuv run motor-thermal evaluate --mode fast\n```"
    )
    st.stop()
mode = st.sidebar.selectbox(
    "Trained models",
    modes,
    format_func=lambda m: {
        "full": "full mode (final results)",
        "fast": "fast mode (preliminary subset)",
    }[m],
)
if mode == "fast":
    st.sidebar.warning("Fast-mode models: trained on a small subset - preliminary results only.")
results = inf.load_results(mode)
manifest = get_manifest()
st.sidebar.markdown("---")
st.sidebar.caption(
    "Educational project (UCS321 mini project #6). The data is the public Paderborn University "
    "LEA motor test-bench set (CC BY-SA 4.0), **not** General Electric hardware. "
    "Not a certified motor-protection system."
)


def need_dataset() -> pd.DataFrame:
    df = get_dataset()
    if df is None:
        st.warning(
            "The recorded measurements are not on this machine. Run `uv run motor-thermal data` "
            "(downloads measures_v2.csv via kagglehub) and reload this page."
        )
        st.stop()
    return df


def line_fig(title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.0, y=0.98, yanchor="top"),
        yaxis_title=ytitle,
        height=440,
        margin=dict(l=10, r=10, t=95, b=10),
        legend=dict(orientation="h", x=0.0, y=1.0, yanchor="bottom"),
        template="plotly_white",
    )
    return fig


def thin(df: pd.DataFrame, max_pts: int = 6000) -> pd.DataFrame:
    return df.iloc[:: max(1, len(df) // max_pts)]


# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Electric-motor winding temperature: a machine-learning virtual sensor")
    st.markdown(
        "**Question.** Can ML estimate the *stator winding* temperature of a permanent-magnet "
        "synchronous motor (PMSM) on **unseen operating profiles**, without access to that "
        "temperature at inference time?"
    )
    c1, c2, c3 = st.columns(3)
    if results:
        c1.metric(
            "Track A · instantaneous · test MAE",
            f"{results['track_A']['test']['aggregate']['mae']:.2f} °C",
        )
        c2.metric(
            "Track B · history-aware · test MAE",
            f"{results['track_B']['test']['aggregate']['mae']:.2f} °C",
        )
        c3.metric("Held-out test profiles", len(results["test_profiles"]))
    arch = FIGURES_DIR / "architecture.png"
    if arch.is_file():
        st.image(str(arch), caption="Pipeline architecture")
    st.subheader("Two inference modes")
    st.markdown(
        "- **Track A - instantaneous:** only the current measurements (speed, torque, d-q currents and "
        "voltages, coolant and ambient temperature). Works for one manual input; knows nothing about "
        "how hot the motor already is.\n"
        "- **Track B - history-aware:** adds *strictly causal* summaries of recent load and coolant "
        "(exponential and rolling averages from 30 s to 2 h). Captures thermal inertia; needs a "
        "running history buffer.\n"
        "- **Never used as inputs:** measured winding, tooth, yoke or magnet temperature, or `profile_id`."
    )
    st.subheader("Limitations")
    st.markdown(
        "- Trained on one test bench; extrapolation to other motors or operating regions is unverified.\n"
        "- Estimates are statistical; typical errors are shown next to every result.\n"
        "- The residual monitor is experimental and does not diagnose causes; there is no labelled fault data.\n"
        "- History features assume the buffer is filled; right after a restart the estimate degrades."
    )
    if manifest:
        st.caption(
            f"Split (whole profiles): train {manifest['train']['n_profiles']} · "
            f"validation {manifest['val']['n_profiles']} · test {manifest['test']['n_profiles']}."
        )

elif page == "Dataset exploration":
    st.title("Dataset exploration")
    df = need_dataset()
    prof_path = ARTIFACTS_DIR / "profile_table.csv"
    prof = pd.read_csv(prof_path, index_col=0) if prof_path.is_file() else None
    pids = sorted(int(p) for p in df[GROUP].unique())
    c1, c2 = st.columns([1, 3])
    pid = c1.selectbox(
        "Operating profile",
        pids,
        index=pids.index(manifest["test"]["profiles"][0]) if manifest else 0,
        format_func=lambda p: f"{p}  ({split_of(p, manifest)})",
    )
    g = df[df[GROUP] == pid].reset_index(drop=True)
    t = np.arange(len(g)) / HZ / 60
    c1.metric("Duration", f"{len(g) / HZ / 3600:.2f} h")
    c1.metric("Peak winding", f"{g[TARGET].max():.1f} °C")
    c1.metric("Start winding", f"{g[TARGET].iloc[0]:.1f} °C")
    fig = line_fig(f"Profile {pid}: temperatures", "°C")
    idx = thin(g).index
    for col, color in ((TARGET, DARK), ("coolant", BLUE), ("ambient", "#6b7280")):
        fig.add_trace(
            go.Scattergl(x=t[idx], y=g.loc[idx, col], name=col, line=dict(color=color, width=1.4))
        )
    c2.plotly_chart(fig, use_container_width=True)
    fig2 = line_fig("Drive signals", "rpm / Nm / A")
    for col, color in (
        ("motor_speed", ORANGE),
        ("torque", GREEN),
        ("i_q", BLUE),
        ("i_d", "#7c3aed"),
    ):
        fig2.add_trace(
            go.Scattergl(x=t[idx], y=g.loc[idx, col], name=col, line=dict(color=color, width=1))
        )
    st.plotly_chart(fig2, use_container_width=True)
    a, b = st.columns(2)
    sample = df.iloc[::25]
    h = go.Figure(go.Histogram(x=sample[TARGET], nbinsx=80, marker_color=BLUE))
    h.update_layout(
        title="Winding temperature distribution (all profiles, thinned)",
        height=340,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    a.plotly_chart(h, use_container_width=True)
    corr_path = ARTIFACTS_DIR / "eda_correlation_train.csv"
    if corr_path.is_file():
        corr = pd.read_csv(corr_path, index_col=0)
        hm = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
            )
        )
        hm.update_layout(
            title="Correlation (training profiles)",
            height=340,
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        b.plotly_chart(hm, use_container_width=True)
    if prof is not None:
        st.subheader("All profiles")
        st.dataframe(prof.round(2), use_container_width=True, height=300)

elif page == "Instantaneous prediction":
    st.title("Instantaneous prediction (Track A)")
    st.markdown(
        '<div class="note"><b>This is a model estimate, not a sensor reading.</b> It uses only the '
        "present measurements, so it cannot know how long the motor has been heating - expect large "
        "errors when the thermal state differs from what the inputs suggest. Use Track B on the "
        "<i>Profile replay</i> page for history-aware estimates.</div>",
        unsafe_allow_html=True,
    )
    sensor = get_sensor(mode, "A")
    ranges = inf.load_input_ranges(mode)
    defaults = {
        "ambient": 24.8,
        "coolant": 27.0,
        "motor_speed": 2000.0,
        "torque": 11.0,
        "i_d": -51.0,
        "i_q": 16.0,
        "u_d": -7.0,
        "u_q": 49.0,
    }
    ds = get_dataset()
    if ds is not None:
        st.caption("Optionally start from a real recorded sample:")
        c1, c2, c3 = st.columns([1, 2, 1])
        pid_ex = c1.selectbox("Profile", sorted(int(p) for p in ds[GROUP].unique()), key="ex_pid")
        gl = int((ds[GROUP] == pid_ex).sum())
        pos = c2.slider("Position in profile [min]", 0.0, gl / HZ / 60, gl / HZ / 120, key="ex_pos")
        if c3.button("Load sample", use_container_width=True):
            row = ds[ds[GROUP] == pid_ex].iloc[min(int(pos * 60 * HZ), gl - 1)]
            for k in inf.RAW_INPUTS:
                st.session_state[f"in_{k}"] = float(row[k])
            st.session_state["ex_measured"] = float(row[TARGET])
    spec = {
        "ambient": ("Ambient temperature [°C]", -10.0, 45.0),
        "coolant": ("Coolant temperature [°C]", 5.0, 105.0),
        "motor_speed": ("Motor speed [rpm]", -1000.0, 6500.0),
        "torque": ("Torque [Nm]", -300.0, 300.0),
        "i_d": ("d-axis current i_d [A]", -320.0, 20.0),
        "i_q": ("q-axis current i_q [A]", -320.0, 320.0),
        "u_d": ("d-axis voltage u_d [V]", -150.0, 150.0),
        "u_q": ("q-axis voltage u_q [V]", -40.0, 150.0),
    }
    cols = st.columns(2)
    sample_in = {}
    for i, (k, (label, lo, hi)) in enumerate(spec.items()):
        key = f"in_{k}"
        if key not in st.session_state:
            st.session_state[key] = defaults[k]
        v = float(np.clip(st.session_state[key], lo, hi))
        st.session_state[key] = v
        sample_in[k] = cols[i % 2].slider(label, lo, hi, key=key)
    warns = inf.check_input_ranges(sample_in, ranges)
    for w in warns:
        st.warning(f"Extrapolation risk: {w}")
    est = sensor.predict_one(sample_in)
    d = inf.instant_features_scalar(sample_in)
    typical = results["track_A"]["test"]["aggregate"] if results else None
    c1, c2 = st.columns([1, 1])
    c1.metric("Estimated winding temperature", f"{est:.1f} °C")
    if typical:
        c1.caption(
            f"Typical error on unseen profiles: MAE {typical['mae']:.1f} °C, 95 % of errors below {typical['p95_ae']:.1f} °C."
        )
    if "ex_measured" in st.session_state and ds is not None:
        c2.metric(
            "Recorded winding temperature of that sample",
            f"{st.session_state['ex_measured']:.1f} °C",
            delta=f"{est - st.session_state['ex_measured']:+.1f} °C estimate error",
            delta_color="off",
        )
    st.subheader("Derived features computed automatically")
    kinds = feat.FEATURE_KIND
    st.dataframe(
        pd.DataFrame(
            {
                "feature": list(kinds),
                "value": [round(d[k], 2) for k in kinds],
                "kind": list(kinds.values()),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

elif page == "Profile replay":
    st.title("Operating-profile replay")
    st.markdown(
        '<div class="note">Replay of <b>recorded historical data</b> from the test bench - not live '
        "monitoring of a physical motor. Estimates are computed causally: each point uses only the "
        "past and present.</div>",
        unsafe_allow_html=True,
    )
    df = need_dataset()
    pids = sorted(int(p) for p in df[GROUP].unique())
    default = manifest["test"]["profiles"][0] if manifest else pids[0]
    c1, c2 = st.columns([2, 1])
    pid = c1.selectbox(
        "Profile",
        pids,
        index=pids.index(default),
        format_func=lambda p: f"{p}  ({split_of(p, manifest)})",
    )
    show_a = c2.checkbox("Show Track A (instantaneous)", value=True)
    if split_of(pid, manifest) == "train":
        st.info(
            "This profile was part of the training data - errors here are optimistic. Pick a *test* profile for an honest view."
        )
    pred = profile_predictions(mode, pid)
    n = len(pred)
    pos = st.slider(
        "Replay position [min]",
        float(pred.t_min.iloc[min(120, n - 1)]),
        float(pred.t_min.iloc[-1]),
        float(pred.t_min.iloc[-1]),
        key=f"pos_{pid}",
    )
    cut = pred[pred.t_min <= pos]
    fig = line_fig(f"Profile {pid}: measured vs. estimated winding temperature", "°C")
    c = thin(cut)
    fig.add_trace(
        go.Scattergl(
            x=c.t_min, y=c.measured, name="measured (reference)", line=dict(color=DARK, width=2)
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=c.t_min, y=c.pred_B, name="Track B - history-aware", line=dict(color=BLUE, width=1.5)
        )
    )
    if show_a:
        fig.add_trace(
            go.Scattergl(
                x=c.t_min,
                y=c.pred_A,
                name="Track A - instantaneous",
                line=dict(color=ORANGE, width=1.1),
            )
        )
    fig.update_xaxes(range=[0, pred.t_min.iloc[-1]])
    st.plotly_chart(fig, use_container_width=True)
    res = cut.measured - cut.pred_B
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Measured now", f"{cut.measured.iloc[-1]:.1f} °C")
    m2.metric(
        "Track B estimate now",
        f"{cut.pred_B.iloc[-1]:.1f} °C",
        delta=f"{cut.pred_B.iloc[-1] - cut.measured.iloc[-1]:+.1f} vs measured",
        delta_color="off",
    )
    m3.metric("Track B MAE so far", f"{res.abs().mean():.2f} °C")
    m4.metric("Track A MAE so far", f"{(cut.measured - cut.pred_A).abs().mean():.2f} °C")
    fr = line_fig("Residual (measured − Track B estimate)", "°C")
    fr.add_trace(
        go.Scattergl(
            x=c.t_min, y=(c.measured - c.pred_B), line=dict(color=DARK, width=1), name="residual"
        )
    )
    fr.add_hline(y=0, line_color="#6b7280")
    fr.update_layout(height=280)
    st.plotly_chart(fr, use_container_width=True)
    st.caption(
        "Rolling-window features need up to 5 min of history, and the slowest exponential averages ~2 h; "
        "early parts of a profile are therefore less accurate (see Model evaluation)."
    )

elif page == "Model evaluation":
    st.title("Model evaluation (held-out profiles)")
    if not results:
        st.warning(f"Run `uv run motor-thermal evaluate --mode {mode}` to create test results.")
        st.stop()
    fig_dir = FIGURES_DIR / mode
    ta, tb = results["track_A"], results["track_B"]
    rows = []
    for name, t in (("Track A - instantaneous", ta), ("Track B - history-aware", tb)):
        a, p = t["test"]["aggregate"], t["test"]["profile_level"]
        rows.append(
            {
                "model": name,
                "candidate": t["candidate"],
                "MAE °C": a["mae"],
                "RMSE °C": a["rmse"],
                "R²": a["r2"],
                "median AE": a["median_ae"],
                "p95 AE": a["p95_ae"],
                "profile-mean MAE": p["macro_mae"],
                "worst profile MAE": p["worst_profile_mae"],
            }
        )
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.caption(
        f"Test = {len(results['test_profiles'])} whole profiles never used for training, tuning or model selection: "
        f"{results['test_profiles']}"
    )
    cand_path = ARTIFACTS_DIR / mode / "candidates_test.csv"
    if cand_path.is_file():
        st.subheader("All candidates")
        cdf = pd.read_csv(cand_path).sort_values("test_mae")
        st.dataframe(
            cdf[
                [
                    "candidate",
                    "n_features",
                    "test_mae",
                    "test_rmse",
                    "test_r2",
                    "test_macro_mae",
                    "test_worst_profile_mae",
                    "fit_seconds",
                    "model_size_kb",
                    "single_row_latency_ms",
                ]
            ].round(3),
            hide_index=True,
            use_container_width=True,
        )
    tabs = st.tabs(
        [
            "Estimate vs measured",
            "Residuals",
            "Per profile",
            "Error by range",
            "Feature importance",
            "History ablation",
            "Case study",
        ]
    )

    def show(tab, *names):
        with tab:
            cols = st.columns(len(names))
            for col, nme in zip(cols, names, strict=True):
                p = fig_dir / nme
                if p.is_file():
                    col.image(str(p))

    show(tabs[0], "scatter_track_b.png", "scatter_track_a.png")
    show(tabs[1], "residuals_track_b.png")
    show(tabs[2], "per_profile_mae.png")
    show(tabs[3], "error_by_temperature_range.png")
    show(tabs[4], "importance_groups_track_b.png", "importance_track_b.png")
    show(tabs[5], "history_ablation.png")
    show(tabs[6], "case_worst_profile.png")
    if "warm_start_robustness_track_B" in results and results["warm_start_robustness_track_B"].get(
        "profiles_used"
    ):
        w = results["warm_start_robustness_track_B"]
        st.info(
            f"Restart robustness: emptying the history buffer after {w['restart_after_s'] / 60:.0f} min on already-warm "
            f"motors raises MAE in the next 30 min from {w['mae_first_30min_after_restart_full_history']:.2f} °C "
            f"to {w['mae_first_30min_after_restart_restarted']:.2f} °C."
        )

elif page == "Thermal monitoring":
    st.title("Thermal residual monitoring (experimental)")
    st.markdown(
        '<div class="note"><b>Not fault detection and not a protection system.</b> A persistent residual '
        "(measured − estimated) may come from a sensor problem, a cooling change, an operating condition "
        "outside the training data, ordinary model error, or a real motor issue. The dataset has no labelled "
        "faults, so no detection accuracy is claimed.</div>",
        unsafe_allow_html=True,
    )
    if not results:
        st.warning("Run the evaluation stage first.")
        st.stop()
    df = need_dataset()
    mcfg = CFG["monitoring"]
    band = results["monitoring"]["test"]["band_smoothed_c"]
    pids = sorted(int(p) for p in df[GROUP].unique())
    c1, c2 = st.columns([2, 1])
    pid = c1.selectbox(
        "Profile",
        pids,
        index=pids.index(results["test_profiles"][0]),
        format_func=lambda p: f"{p}  ({split_of(p, manifest)})",
    )
    offset = c2.slider(
        "Synthetic sensor offset added to the measurement [°C]",
        0.0,
        20.0,
        0.0,
        0.5,
        help="Illustration of sensitivity only - synthetic, not a real fault.",
    )
    pred = profile_predictions(mode, pid)
    meas = pred.measured.to_numpy() + offset
    res = meas - pred.pred_B.to_numpy()
    pids_arr = np.zeros(len(res), dtype=int)
    window, persist = int(mcfg["smoothing_window_s"] * HZ), int(mcfg["persistence_s"] * HZ)
    sm = mon.causal_smooth(res, pids_arr, window)
    flags = mon.persistent_flags(sm, pids_arr, band, persist)
    raw_out = np.abs(res) > band
    fig = line_fig("Measured vs estimated", "°C")
    c = thin(pred, 5000).index
    fig.add_trace(
        go.Scattergl(
            x=pred.t_min[c],
            y=meas[c],
            name="measured" + (f" (+{offset:g} °C synthetic)" if offset else ""),
            line=dict(color=DARK, width=1.8),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=pred.t_min[c],
            y=pred.pred_B[c],
            name="estimated (Track B)",
            line=dict(color=BLUE, width=1.3),
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    fr = line_fig(
        f"Smoothed residual ({mcfg['smoothing_window_s']} s) with ±{band:.1f} °C band calibrated on validation profiles",
        "°C",
    )
    fr.add_hrect(y0=-band, y1=band, fillcolor=GREEN, opacity=0.12, line_width=0)
    fr.add_trace(
        go.Scattergl(
            x=pred.t_min[c], y=sm[c], name="smoothed residual", line=dict(color=DARK, width=1.4)
        )
    )
    if flags.any():
        fr.add_trace(
            go.Scattergl(
                x=pred.t_min[c][flags[c]],
                y=sm[c][flags[c]],
                mode="markers",
                name="persistent alarm",
                marker=dict(color=RED, size=5),
            )
        )
    st.plotly_chart(fr, use_container_width=True)
    a, b, d = st.columns(3)
    a.metric("Samples with isolated |residual| > band", f"{raw_out.mean() * 100:.1f} %")
    b.metric(
        f"Persistent alarm (> {mcfg['persistence_s']} s)", f"{flags.mean() * 100:.1f} % of time"
    )
    d.metric("Alarm triggered", "yes" if flags.any() else "no")
    mt = results["monitoring"]
    st.subheader("Behaviour on the held-out test profiles (healthy data, no injected offset)")
    st.markdown(
        f"Band ±{mt['test']['band_smoothed_c']:.1f} °C = {mt['band_quantile'] * 100:.1f}th percentile of |smoothed residual| on "
        f"**validation** profiles. On the test profiles: {mt['test']['smooth_exceed_frac'] * 100:.2f} % of samples exceed it, "
        f"{mt['test']['persistent_alarm_frac'] * 100:.2f} % of time is in persistent alarm, "
        f"{mt['test']['profiles_with_any_alarm']} of {mt['test']['n_profiles']} profiles raise at least one alarm. "
        "These are false alarms or model error - empirical bands are not probabilistic guarantees."
    )
    inj = pd.DataFrame(mt["synthetic_injected_offset_on_test"])
    st.caption("Synthetic injected-offset sensitivity on the test profiles (illustration only):")
    st.dataframe(inj.round(2), hide_index=True, use_container_width=True)
