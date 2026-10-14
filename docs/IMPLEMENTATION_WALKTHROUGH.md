# Implementation walkthrough

A file-by-file explanation of the repository, the order in which things run, the important functions and
formulas, and the life of a model from raw CSV to prediction. Concepts (why we do things) are in
`LEARN_THE_PROJECT.md`; this document is about **where and how** the code does them.

---

## 1. Repository map

```
configs/default.yaml        every experiment setting (windows in seconds, split, search space, thresholds)
src/motor_thermal/
  config.py                 load/validate YAML, project paths, config hash
  data.py                   find/download the CSV, load as compact dtypes, validate schema, provenance
  splits.py                 deterministic whole-profile train/val/test manifest + guarantees
  features.py               instantaneous features, causal history features (offline + streaming)
  metrics.py                MAE/RMSE/R²/percentiles, per-profile, per-bin, per-region tables
  models.py                 estimator factory (sklearn Pipelines), fit/predict/benchmark, save/load bundles
  train.py                  candidate comparison, history ablation, tuning, group-CV, frozen selection
  evaluate.py               the only place that reads TEST profiles; plots, importance, monitoring
  monitor.py                residual smoothing, persistence alarm, band calibration, offset sensitivity
  forecast.py               optional 60 s-ahead forecasting experiment (separate from the estimator)
  eda.py                    exploratory plots and summary (train profiles drive anything modelling-related)
  plots.py                  all matplotlib figures incl. the architecture diagram
  inference.py              VirtualSensor / StreamingPredictor used by the app and tests
  report.py, report_docs.py fill the markdown templates (src/motor_thermal/templates/) with real results
  deck.py                   editable PowerPoint (python-pptx) built from the same result files
  cli.py                    `motor-thermal` command-line entry point
app/streamlit_app.py        the dashboard
tests/                      pytest suite (synthetic toy data only for code behaviour)
scripts/run_pipeline.sh     reproduce everything
artifacts/                  models, metrics, split manifest, provenance (small, committed)
reports/                    generated figures, report, presentation
docs/                       this guide, learning guide, viva preparation
```

---

## 2. Execution sequence

```
uv sync                                   # create .venv from pyproject.toml / uv.lock (Python 3.12)
uv run motor-thermal data                 # locate or download CSV, validate, write provenance.json
uv run motor-thermal eda                  # figures/eda/*, reports/eda_summary.json
uv run motor-thermal train    --mode full # candidates, ablation, tuning, CV, selection.json, models
uv run motor-thermal evaluate --mode full # TEST metrics, plots, importance, monitoring
uv run motor-thermal forecast --mode full # optional 60 s-ahead experiment
uv run motor-thermal report   --mode full # report.md, viva doc, architecture diagram, pptx
uv run streamlit run app/streamlit_app.py
```

`--mode fast` runs the same code on half of the training profiles with a coarser row stride (a smoke
test that gives labelled preliminary numbers in a couple of minutes). `--csv PATH` points at a manually
downloaded file.

---

## 3. Module details

### 3.1 `config.py`
`load_config()` reads the YAML and refuses configs where an input column is also on the forbidden list.
`config_hash()` is a short SHA-256 of the canonical JSON of the config; it is stored in every model's
metadata so a result can be tied to the exact settings. `mode_dir(mode)` returns `artifacts/<mode>`.

### 3.2 `data.py`
- `find_local_dataset` - looks (without network) at `--csv`, `$MOTOR_THERMAL_CSV`, `data/measures_v2.csv`
  and the kagglehub cache. An explicit path is exclusive (no silent fallback).
- `locate_dataset` - same, then `kagglehub.dataset_download` (the dataset is public; anonymous access
  worked). On failure it raises `DatasetUnavailableError` with manual instructions; the CLI exits with
  code 2 instead of fabricating data.
- `load_measurements` - reads the CSV with `float32`/`int16` to save memory and caches a parquet.
  **Row order is preserved**: the file has no timestamps and row order within a profile is the time axis.
- `validate_schema` - required columns, NaN count, documented row/profile counts (warnings), physical
  range checks (warnings), duplicate rows and **profile contiguity** (each `profile_id` must be one
  unbroken block; otherwise row order cannot be time and validation *fails*).
- `profile_table`, `build_provenance` - per-profile summary; source URL, licence, file size, SHA-256,
  rows, profiles, columns, assumed sampling rate.

### 3.3 `splits.py`
`make_split_manifest` sorts profiles by peak winding temperature, cuts them into chunks of
`chunk_size` (7) and lets each chunk give `val_per_chunk`/`test_per_chunk` profiles by a seeded
permutation. `assert_manifest_valid` enforces: splits disjoint, together equal to all profiles, none
empty. `split_frame` selects rows by `profile_id` membership (it never touches individual rows).

### 3.4 `features.py`
**Instantaneous** (`add_instant_features`) - formulas:

```
i_sq = i_d² + i_q²                 i_mag = √i_sq              u_mag = √(u_d² + u_q²)
abs_speed = |ω| (rpm)              abs_torque = |T| (N·m)
p_mech_w = T · ω · 2π/60           (exact, if N·m and rpm)
p_elec_proxy = 1.5 (u_d i_d + u_q i_q)   (proxy - convention-dependent)
p_loss_proxy = p_elec_proxy − p_mech_w   (proxy for total loss)
ambient_minus_coolant = T_amb − T_cool
```
`FEATURE_KIND` documents which are exact and which are proxies; the app shows it.

**History** (`add_history_features`) works per profile on contiguous blocks:

```
EWMA with time constant τ [s]:   α = 1 − exp(−Δt/τ),   Δt = 1/hz
    pandas ewm(alpha=α, adjust=True) → value_k = Σ (1−α)^(k−j) x_j / Σ (1−α)^(k−j)   (defined from k=0)
rolling mean over W seconds:      n = W·hz samples, min_periods = n   → NaN for the first n−1 samples
```

`StreamingHistory` keeps `num = d·num + x`, `den = d·den + 1` (so `num/den` equals the same EWMA) and a
ring buffer for the rolling means. `reset()` starts a new profile. Tests compare both versions.

`feature_list(cfg, "enhanced", set_name)` returns the ordered input columns of a model and calls
`assert_no_forbidden` (rejects `stator_*`, `pm*`, `profile_id`). The *superset* frame
(`ewma_long_rolling`) is computed once; every other history set is a column subset.

### 3.5 `metrics.py`
`regression_metrics` (MAE, RMSE, R², median/p95/max AE, bias), `per_profile_metrics`,
`summarize_profiles` (macro MAE, worst profile), `error_by_bins` (temperature ranges),
`operating_region_table` (3×3 speed × torque terciles taken from **training** data), `summarize_all`.

### 3.6 `models.py`
`make_estimator(name, ...)` returns a scikit-learn `Pipeline`:

| name | pipeline |
|---|---|
| `mean_baseline` | `DummyRegressor(mean)` |
| `linear`, `ridge` | `StandardScaler` → `LinearRegression` / `Ridge` |
| `random_forest` | `RandomForestRegressor` (trained on a bounded random sample) |
| `hist_gb` | `HistGradientBoostingRegressor` (fixed iterations; sklearn's internal early stopping uses a random row split, which is leaky for time series) |
| `lightgbm` | `LGBMRegressor` (early stopping on the validation profiles, MAE metric) |

`fit_candidate` times the fit and returns a `FitRecord`; `benchmark_inference` measures batch throughput
and single-row latency; `save_model_bundle`/`load_model_bundle` write/read `track_x.joblib` + `track_x.json`.

### 3.7 `train.py` - the training stage
1. `load_everything`: locate → load → validate (abort on errors) → profile table → manifest → full-rate
   feature frame.
2. `thin_training_rows`: optional seeded subset of training profiles (fast mode) and a row stride **after**
   features were computed; returns a record of what was kept.
3. Track A candidates: baseline, physical linear (`coolant`, `i_sq`), and {Linear, Ridge, RF, HistGB,
   LightGBM} × {basic, enhanced} features.
4. History ablation: LightGBM with default hyperparameters on each history set, repeated with several
   seeds (`seed_val_maes`); the set with the best mean validation MAE wins.
5. HistGB on the winning history set, then a seeded random search of LightGBM hyperparameters in both
   tracks. **Each trial has a unique id** (`...[t3]...`); an early version overwrote the record and saved
   the wrong "winner" - `test_candidate_ids_are_unique_and_saved_winner_is_the_validated_one` guards this.
6. Group k-fold CV (`GroupKFold`) on train+val profiles for the headline configurations (full mode).
7. Frozen selection: the lowest validation MAE inside each track → `selection.json`, with the note that
   **test profiles were not read**. All candidates go to `models_all/` (git-ignored), winners to `models/`.
8. Writes `candidates_val.csv`, `history_ablation_val.csv`, `lgbm_search_log.json`, `group_cv.csv`,
   `run_record.json` (seed, config hash, environment, durations, peak memory, subset info).

### 3.8 `evaluate.py` - the evaluation stage
Refuses to run without `selection.json`. Verifies the regenerated split equals the saved manifest. Then:
test metrics for **every** candidate in one pass (`candidates_test.csv`); for both winners: aggregate and
profile-level metrics, error by temperature range, operating region and elapsed time in profile,
inference benchmark; permutation importance (single features and feature *families*) on a bounded
**validation** sample; restart-robustness experiment (empty the history buffer after 1 h on warm motors);
residual monitoring; figures; `metrics_test.json`.

### 3.9 `monitor.py`
```
smoothed = causal rolling mean (window = smoothing_window_s · hz) of (measured − predicted), per profile
band     = quantile_q(|smoothed|) on VALIDATION profiles
alarm_k  = 1 once |smoothed| > band has held for ≥ persistence_s samples (run counter resets otherwise)
```
`injected_offset_sensitivity` adds a constant synthetic bias to the measurement and reports how many
profiles alarm and how fast - an illustration of sensitivity, not a detection accuracy.

### 3.10 `inference.py`
`VirtualSensor.load(mode, track)` loads model + metadata. `predict_frame` computes the same features as
training (history per single profile) and predicts; `predict_one` is the Track A single-sample path
using `instant_features_scalar`; `stream()` returns a `StreamingPredictor` (sample-by-sample). Models only
ever receive the columns listed in their metadata, so measured temperatures in the input frame cannot
influence predictions (tested by corrupting them).

### 3.11 `app/streamlit_app.py`
Sidebar chooses the model set (`full` preferred) and page. `st.cache_resource` caches loaded models,
`st.cache_data` caches the dataset and per-profile predictions. If models are missing the app shows how to
train them; if the dataset is missing the replay pages show how to fetch it. Nothing is simulated.

### 3.12 `report.py`, `report_docs.py`, `deck.py`
`report.py` / `report_docs.py` read `metrics_test.json`, the CSV tables and EDA summary, fills the markdown templates in
`src/motor_thermal/templates/`, draws the architecture diagram and calls `deck.build_deck`. Every
number in the report and slides is read from those files.

---

## 4. Model life cycle

```
CSV ─► validate ─► split (profiles) ─► features (2 Hz, causal) ─► thin rows ─► fit candidates
     ─► validation compare/ablate/tune ─► select (frozen) ─► save bundle (.joblib + .json)
     ─► test (once) ─► plots/metrics ─► app loads bundle ─► predict (batch / one / streaming)
```

## 5. Tests (what each file guarantees)

| file | guarantee |
|---|---|
| `test_config_data.py` | config sanity & stable hash; schema, NaN, **contiguity** checks; dtypes, parquet cache |
| `test_splits.py` | exclusivity, completeness, determinism, 49/10/10 layout, manifest validation catches errors |
| `test_features.py` | formulas; forbidden features rejected; **causality**; **profile reset**; **cold-start NaN**; EWMA closed form; **streaming == offline** |
| `test_metrics_monitor.py` | metric values; profile-level vs aggregate; smoothing/persistence/band logic; sensitivity logic |
| `test_pipeline_e2e.py` | full train → select → evaluate → artifacts → inference on synthetic data; test profiles unseen by training; **leakage invariance**; unique candidate ids; refusal to evaluate before selection |

Run everything: `uv run pytest -q`; lint: `uv run ruff check . && uv run ruff format --check .`.
