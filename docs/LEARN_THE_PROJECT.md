# Learn the project: from first principles to the code

This guide is written for a second-year Electronics & Computer Engineering student. It explains the
ideas behind the project in the order you need them. Read it top to bottom once, then keep it open while
you read the code (Part 12 maps every idea to a file). Exact results are **not** repeated here; they live
in `reports/project_report.md` and `artifacts/full/metrics_test.json`, which are generated from real runs.

---

## 1. The problem in one paragraph

A permanent-magnet synchronous motor (PMSM) heats up when it works. If the **stator winding** (the copper
coils) gets too hot, the insulation ages quickly and the motor can fail, so the controller must limit
torque near the thermal limit. Putting a temperature sensor *inside* the winding is costly, fragile and
hard to do in series production. Idea: **estimate** the winding temperature from signals the drive already
measures (speed, torque, currents, voltages, coolant and ambient temperature). A model that does this is
called a **virtual sensor** or *soft sensor*.

The central question of the project: *can a model learn this estimate well enough to work on operating
profiles it has never seen?* "Never seen" matters: a model that only works on data resembling its training
data is useless in the field.

---

## 2. Supervised learning vs. conventional programming

**Conventional programming:** you write the rule. `if temperature > 120: reduce_torque()`.

**Supervised learning:** you give the computer examples of inputs with the correct answers and it finds
the rule.

```
examples:  (speed, torque, currents, ...)  ->  measured winding temperature
learning:  find a function f such that f(inputs) ≈ temperature on the examples
use:       f(new inputs) -> estimated temperature
```

Why not write the physics by hand? You can (a *thermal network* of resistances and capacitances), but it
needs many motor-specific parameters that are hard to measure and change with temperature and ageing.
Machine learning needs no parameters other than data, at the cost of needing good data and giving no
guarantees outside it. This project also *borrows from physics*: our features are chosen because physics
says they matter (Part 8).

### Vocabulary

| Term | Meaning in this project |
|---|---|
| **Sample / row** | one measurement instant (every 0.5 s) |
| **Features** (inputs, X) | numbers the model sees: speed, torque, `i_d`, `i_q`, `u_d`, `u_q`, coolant, ambient, plus derived ones |
| **Target** (label, y) | what we want to predict: `stator_winding` temperature |
| **Parameters** | numbers the model *learns* (weights of a linear model, split thresholds and leaf values of a tree) |
| **Hyperparameters** | settings *you* choose before learning (tree depth, learning rate, number of trees) |
| **Training** | finding the parameters from the training data |
| **Inference** | using the trained model on new inputs |

---

## 3. Regression vs. classification

- **Classification** predicts a category ("healthy / faulty", "cat / dog").
- **Regression** predicts a number on a continuous scale ("86.4 °C").

Winding temperature is a number, so this is regression. The size of an error matters (being 2 °C off is
better than being 20 °C off), which is why we measure errors in degrees Celsius.

(The residual *monitor* in Part 13 draws a yes/no conclusion, but it is built **on top of** the regression
model and is not itself a trained classifier.)

---

## 4. Linear regression, mathematically

Model: `ŷ = w₁x₁ + w₂x₂ + … + w_p x_p + b`, or `ŷ = Xw` if we append a column of ones to `X`.

Learning = choose `w` to minimise the **mean squared error**

```
L(w) = (1/n) Σ (yᵢ − ŷᵢ)²
```

Setting the gradient to zero gives the **normal equation** `w = (XᵀX)⁻¹ Xᵀy` (in practice solved with
numerically stable factorisations). Alternatively gradient descent repeats `w ← w − η ∇L(w)`.

**Ridge regression** adds a penalty `λ‖w‖²`, which shrinks weights and stabilises the solution when
features are strongly correlated (ours are: `i_sq`, `i_mag` and the power proxies all move together).

**Why scale features?** Ridge penalises large weights, and a feature measured in thousands (rpm) would be
treated differently from one measured in tens (A) purely because of units. `StandardScaler` rescales every
feature to mean 0, std 1. Important: the mean and std are **learned from the training data only** and then
reused for validation/test data (otherwise information leaks - Part 9). Putting scaler + model into one
scikit-learn `Pipeline` makes that automatic.

**Limitation:** a straight line cannot express "heating depends on the *square* of current, saturates, and
depends on how long the load lasted". Our baseline `A/linear/physical` uses only `coolant` and `i_sq`
(the textbook steady-state relation `T ≈ T_coolant + R_th · I²`) and is therefore a meaningful baseline,
not a straw man.

---

## 5. Decision trees and Random Forests

A **decision tree** repeatedly asks yes/no questions about one feature ("is `coolant` < 31 °C?") to split
the data into groups; each final group (a *leaf*) predicts the **average target** of its training samples.
Splits are chosen greedily to reduce the squared error the most.

- Trees capture non-linear effects and interactions automatically and need no scaling.
- A deep tree memorises the training data (overfits).
- Trees **cannot extrapolate**: outside the training range a tree just returns the value of its last leaf.
  This is a big deal for temperatures (Part 15).

A **Random Forest** trains many trees on random subsets of rows and features and averages them. Averaging
reduces variance (the instability of single trees). It is strong and needs little tuning, but the model
is large (we measured ≈ 100 MB for our forest) and slower to evaluate, so we train it on a bounded sample.

---

## 6. Gradient boosting and LightGBM

Boosting builds trees **sequentially**, each correcting the errors of the ensemble so far:

```
F₀(x) = mean(y)
repeat for m = 1..M:
    residuals rᵢ = yᵢ − F_{m−1}(xᵢ)        (the negative gradient of the squared loss)
    fit a small tree h_m to the residuals
    F_m(x) = F_{m−1}(x) + η · h_m(x)         (η = learning rate, e.g. 0.05)
```

The learning rate `η` makes each step small, which generalises better; the price is needing more trees.

**LightGBM** is a fast, memory-efficient implementation. Key ideas:

- **Histogram binning:** continuous features are bucketed into ≤ 255 bins, so finding a split costs a
  scan over bins, not over all rows.
- **Leaf-wise growth:** it always splits the leaf with the largest loss reduction (so trees are lopsided
  but efficient); `num_leaves` is therefore the main complexity knob.
- **Native handling of missing values (NaN):** each split learns which side NaNs should go to. We exploit
  this for the "cold start" of rolling-window features (Part 8).
- **Regularisation:** `min_child_samples` (minimum samples per leaf), `reg_lambda` (L2 on leaf values),
  `subsample` (row sampling) and `colsample_bytree` (feature sampling).

**Early stopping:** while adding trees, watch the error on *validation* profiles. When it has not improved
for `early_stopping_rounds` (100) consecutive trees, stop and keep the best iteration. This chooses the
number of trees automatically and prevents overfitting.

**Loss function:** the thing being minimised during training - here squared error (L2). We *report*
MAE (an interpretable average error in °C) and also monitor it with `eval_metric="l1"`.

**Why LightGBM instead of deep learning?** The data are tabular, 1.3 M rows and a few dozen features;
gradient-boosted trees are usually as accurate as neural networks on such data, train in seconds on a
laptop CPU, are small and fast (milliseconds per prediction), and are easier to explain. Sequence neural
networks (LSTM/TCN) are legitimate alternatives in the literature, but they need GPUs, tuning and more
code, and our history features already give a tree model the memory it needs.

---

## 7. Overfitting, underfitting, generalisation

- **Underfitting:** model too simple, error high on training *and* new data (the straight-line baseline).
- **Overfitting:** model memorises training quirks, error low on training data, high on new data.
- **Generalisation:** performance on data the model has not seen. That is the only thing we care about.

The **bias-variance** view: simple models have high bias (systematic error), flexible models have high
variance (sensitivity to the particular training set). Regularisation, early stopping, more data and
averaging all try to balance them.

---

## 8. PMSM fundamentals and the physics behind our features

### 8.1 d-q quantities

A three-phase motor has three sinusoidal currents. The **Park transform** converts them into a frame that
rotates with the rotor, in which they become two (almost) constant numbers:

- **`i_d` (direct axis):** current along the rotor magnet flux. It mostly weakens or strengthens the
  magnet field ("field weakening" at high speed makes it negative).
- **`i_q` (quadrature axis):** current perpendicular to it. It produces most of the **torque**
  (`torque ≈ 1.5 · p · ψ · i_q + …`; our data confirm a correlation of ≈ 0.996 between `torque` and `i_q`).
- `u_d`, `u_q` are the corresponding voltages.

Total current magnitude is `|i| = √(i_d² + i_q²)`. The sign and scaling of `i_d`, `i_q`, `u_d`, `u_q`
depend on the transform convention (amplitude- or power-invariant), which the dataset does not state.
Therefore `p_elec_proxy = 1.5·(u_d i_d + u_q i_q)` is labelled a **proxy**. The mechanical power
`torque · speed · 2π/60` is exact **if** torque is in N·m and speed in rpm (value ranges suggest so).

### 8.2 Where the heat comes from

| Loss | Roughly proportional to | Our feature |
|---|---|---|
| Copper (I²R) loss in the winding | `|i|²` (and R grows with temperature) | `i_sq` |
| Iron loss (eddy, hysteresis) | speed² … speed, flux | `abs_speed`, `u_mag` |
| Friction/windage | speed | `abs_speed` |

### 8.3 Thermal inertia - the single most important idea

A body with heat capacity `C` [J/K] connected to its coolant through a thermal resistance `R` [K/W] and
heated by power `P`:

```
C · dT/dt = P(t) − (T − T_coolant)/R          time constant  τ = R·C
```

Discretised with step `Δt` (0.5 s):

```
T[k+1] = (1 − Δt/τ)·T[k] + (Δt/τ)·(T_coolant[k] + R·P[k])
```

This is an **exponentially weighted moving average (EWMA)** of the quantity `T_coolant + R·P` with weight
`α = Δt/τ`. In words: *the temperature is a smoothed version of the heating history; it forgets old
load with time constant τ*.

Consequences:

1. The same instantaneous currents can correspond to very different temperatures depending on what
   happened in the last minutes or hours. A model that only sees the present sample (our **Track A**)
   cannot know this - that is why it has large errors even with a flexible algorithm.
2. A motor has several thermal masses (winding: seconds-minutes; stator iron, housing: minutes-hours), so
   we use several time constants: 30 s, 2 min, 10 min, 30 min and 2 h.
3. If we give the model EWMAs of `i²`, speed, torque and coolant temperature at those time constants,
   it receives (approximately) the states of this thermal system - **Track B**.

### 8.4 Why coolant history matters

The coolant temperature is a measured, slow-moving signal that reflects the thermal state surrounding the
winding. Its history therefore tells the model how "soaked" the machine already is. Caveat: on a test bench
the coolant is controlled by the operator; in a vehicle its dynamics differ, so this relation may not
transfer to another installation.

---

## 9. Data leakage, scaling and imputation

**Data leakage** = information from outside the training situation sneaking into training or evaluation,
making results look better than they are. Typical forms and how we prevent them:

| Leak | Why it is wrong | Our protection |
|---|---|---|
| Using `stator_tooth`, `stator_yoke`, `pm` or `stator_winding` as input | These are measured *temperatures of the same machine* - highly collinear with the target and unavailable in the intended use case | `forbidden_features` in config; `assert_no_forbidden`; tests |
| Using `profile_id` as feature | It identifies *which recording* the row came from, not physics | excluded; used only for grouping |
| Random row split of a time series | Neighbouring rows (0.5 s apart) are almost identical, so a test row has a near-duplicate in training | split by **whole profile** (`splits.py`) |
| Fitting scaler / statistics on all data | Test-set statistics influence training | scaler lives inside the pipeline, fitted on training rows |
| Using future samples in a rolling window | At inference you do not have the future | causal features: value at `t` uses samples ≤ `t` only (test: change the future, past features unchanged) |
| Tuning on the test set | The score becomes optimistic | test profiles are read only by `evaluate.py`, after `selection.json` is frozen |

**Imputation** (filling missing values): the raw dataset has no missing values. The only NaNs are the
*deliberate* cold-start values of rolling windows (e.g. a 5-min rolling mean does not exist during the
first 5 min). We do **not** fill them with zero or the mean, because that invents information; LightGBM
handles NaN natively. The EWMA features are defined from the first sample onwards (normalised weights).

---

## 10. Train / validation / test and group-aware validation

- **Training set:** the model fits its parameters here.
- **Validation set:** you compare models, choose hyperparameters and the early-stopping point here.
- **Test set:** used **once**, at the end, to report honest performance.

Because consecutive samples are near-duplicates, all splits are made by **operating profile** (one
continuous recording = one `profile_id`). The 69 profiles are sorted by peak winding temperature, cut into
chunks of 7 and each chunk contributes 1 validation and 1 test profile (seeded random choice) so that mild
and severe profiles appear in every split. The result is saved to `artifacts/split_manifest.json`.

**Group k-fold CV** (`GroupKFold`) repeats the idea: each fold holds out whole profiles. It gives a
mean ± std of the error, showing how much it depends on *which* profiles are held out (a lot, in this
dataset).

Two ways to average error (both reported):

- **Sample-weighted** (aggregate): long profiles dominate.
- **Profile-level** (macro): each profile counts once; the worst profile shows the worst-case behaviour.

---

## 11. Metrics

- **MAE** = mean of `|ŷ − y|`. Average error in °C. Easy to explain, robust. Our headline metric.
- **RMSE** = `√mean((ŷ − y)²)`. Penalises large errors more; always ≥ MAE.
- **R²** = `1 − Σ(y − ŷ)² / Σ(y − ȳ)²`. Fraction of variance explained; 0 = no better than predicting
  the mean, 1 = perfect.
- **Median AE / p95 AE:** typical and "bad-day" errors.

**Why a high R² can be misleading:** if the temperature varies between 20 and 140 °C, even a model that is
10 °C off on average can have R² ≈ 0.8. R² says nothing about an error in °C, which is what matters for
thermal protection. Always read R² together with MAE/p95 and per-profile results.

---

## 12. How the code is organised and how data flows

```
measures_v2.csv ──► data.py ──► (validate, provenance, profile table)
                        │
                        ▼
                    splits.py ──► split_manifest.json   (whole profiles → train/val/test)
                        │
                        ▼
                  features.py ──► instantaneous features + causal history features (full 2 Hz rate)
                        │
                        ▼
                    train.py ──► fits candidates (models.py), ablation, tuning, group-CV
                        │          freezes selection.json (validation only), saves models
                        ▼
                  evaluate.py ──► final test metrics (metrics.py), plots (plots.py),
                        │          importance, restart robustness, residual monitor (monitor.py)
                        ▼
              inference.py ◄──── app/streamlit_app.py   (loads artifacts, estimates, replays)
                        │
                        ▼
                    report.py ──► reports/project_report.md, figures, PowerPoint
```

Read the modules in this order: `config.py` → `data.py` → `splits.py` → `features.py` → `models.py` →
`train.py` → `evaluate.py` → `monitor.py` → `inference.py` → `app/streamlit_app.py`. A file-by-file
description is in `docs/IMPLEMENTATION_WALKTHROUGH.md`.

---

## 13. Rolling windows, EWMA and cold starts

A **rolling mean** over a window of `n` samples averages the last `n` values. An **EWMA** with time constant
`τ` gives weight `exp(−age/τ)` to a sample `age` seconds old. Both are *causal* if they only look back.

Implementation details that matter:

- Windows are specified in **seconds** in `configs/default.yaml` and converted with `sampling_hz = 2`
  (e.g. 300 s = 600 samples). Thinning the *training rows* (taking every 2nd row) happens **after** the
  features are computed, so the windows keep their physical meaning.
- Features are computed **per profile**; the first value of a new profile must not "remember" the
  previous profile. (A test checks this.)
- `StreamingHistory` is the same calculation done sample-by-sample, as a live system would. A test checks
  that it matches the vectorised version.
- A profile can start with the motor already hot (some do: up to ≈ 99 °C). The history buffer then starts
  empty while the motor is warm, which hurts accuracy; we measure this ("restart robustness").

---

## 14. Residual monitoring and anomaly uncertainty

`residual = measured − estimated`. If the estimator is good, residuals are small. If a residual is large
**and persistent**, something differs from what the model learned.

Our recipe (all numbers in `configs/default.yaml`):

1. smooth the residual with a causal 60 s rolling mean,
2. take the 99.5th percentile of its magnitude on the *validation* profiles as the band,
3. raise an alarm when the smoothed residual stays outside the band for ≥ 120 s.

Why this is **not** fault detection: a persistent residual can be caused by a failing sensor, changed
cooling, an operating point the model never saw, ordinary model error, or a real motor defect - the method
cannot tell them apart, and the dataset has no labelled faults, so we cannot claim a detection rate. We
report how often **healthy** held-out profiles trigger the alarm (false alarms) and illustrate sensitivity
with a **synthetic** injected offset. The band is empirical; it is not a probabilistic guarantee. This is
not a certified protection function.

---

## 15. Limits of extrapolation

Tree ensembles predict averages of training targets; they cannot produce a value outside the range they
saw, and their behaviour in unexplored regions (e.g. a coolant temperature never used in training) is
arbitrary. `inference.check_input_ranges` warns in the app when an input leaves the training span.
Likewise the model is specific to this motor and test bench.

---

## 16. Model serialisation and deployment

`joblib.dump(pipeline, "track_b.joblib")` stores the whole fitted pipeline. A JSON file next to it stores
the **recipe**: ordered feature columns, history windows, hyperparameters, seed, config hash, dataset
checksum and library versions. Loading the model without that recipe would silently produce wrong features,
so `inference.VirtualSensor.load` always reads both. Pickled models should only be loaded from trusted
sources and with the same library versions.

---

## 17. Exercises

1. In `configs/default.yaml` remove the 7200 s time constant and re-run `train --mode fast`. How does
   the validation MAE of Track B move?
2. Write a test that fails if `stator_tooth` is added to `inputs`. (Look at `test_features.py`.)
3. Plot `ewm_i_sq_120s` against `i_sq` for one profile and explain the lag you see.
4. Change `band_quantile` to 0.9 and observe the false-alarm rate on the test profiles.
5. Explain, using the discretised RC equation, why increasing `τ` makes the EWMA respond more slowly.
6. Remove `coolant` from `history_signals`. What happens, and what does that tell you about what the
   model uses?
