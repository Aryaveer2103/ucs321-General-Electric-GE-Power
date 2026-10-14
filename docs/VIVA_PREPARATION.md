# Viva preparation

34 questions with model answers, from basic to challenging. Numbers below are generated from the real run
(**full (final results)** results, `artifacts/full/`); if you re-train, re-run `uv run motor-thermal report`.
Learn the *reasoning*, not just the numbers - examiners test understanding by changing the question.

---

## A. Basics

**1. Summarise the project in two sentences.**
We train a machine-learning model that estimates the stator-winding temperature of a PMSM from signals the
drive already measures (speed, torque, d-q currents/voltages, coolant and ambient temperature), and we test
it only on operating profiles it has never seen. We compare an instantaneous model with a history-aware one,
and demonstrate a residual monitor and a dashboard.

**2. Why not simply install a temperature sensor in the winding?**
Sensors inside the winding cost money, need insulation-safe wiring, add assembly steps, can fail or drift and are
hard to replace once the stator is potted. A virtual sensor needs no extra hardware, can act as a redundant channel
and can also cross-check a physical sensor. The price is that it is a statistical estimate (here typically
2.35 °C off) and valid only inside the conditions it was trained on.

**3. Why is this a regression problem and not classification?**
The output is a continuous quantity (°C) and the size of the error matters. Classification predicts categories
such as "healthy/faulty". Our monitor makes a yes/no decision, but it is built on top of the regression residual
and is not a trained classifier.

**4. What are the features and the target?**
Target: `stator_winding`. Features: `ambient`, `coolant`, `motor_speed`, `torque`, `i_d`, `i_q`, `u_d`, `u_q`, nine derived
physics features (e.g. $i_d^2+i_q^2$, mechanical power) and - for Track B - causal history features (EWMAs and rolling
means of $i^2$, speed, torque and coolant temperature). Measured stator tooth/yoke/magnet temperatures and `profile_id`
are never features.

**5. Parameters vs. hyperparameters?**
Parameters are learned from data (tree split thresholds, leaf values, linear weights). Hyperparameters are chosen by
us before training (number of leaves, learning rate, min samples per leaf, EWMA time constants). We tuned LightGBM's
with a small random search (14 trials per track) scored on validation profiles.

**6. What is inference?**
Using the trained model on new inputs to produce an estimate, as the dashboard does. Single-row latency here is about
3.5 ms for Track B.

**7. Supervised learning vs. conventional programming?**
In conventional programming we write the rule; in supervised learning we provide input-output examples and the
algorithm finds the rule (function) that maps one to the other. Here we do not know a precise formula for winding
temperature because it depends on motor-specific thermal parameters and history.

---

## B. Motor and physics

**8. What are d-axis and q-axis currents?**
The Park transform rewrites the three phase currents in a frame rotating with the rotor. $i_d$ flows along the magnet
flux (mostly field strengthening/weakening, negative at high speed), $i_q$ is perpendicular and produces most of the
torque (corr(torque, $i_q$) = 0.997 in the data). Copper losses scale with $i_d^2 + i_q^2$.

**9. What is thermal inertia and why does it matter here?**
Heat capacity makes temperature respond slowly: $C\,dT/dt = P - (T - T_{cool})/R$. The temperature is thus a smoothed
(low-pass-filtered) function of past losses, and identical present currents can mean very different temperatures.
A model that only sees the present sample cannot know the thermal state; that is why Track A is limited to
≈ 9.53 °C MAE.

**10. How did you encode thermal inertia, and why several time constants?**
With causal EWMAs, $\mathrm{EWMA}_\tau$ (30 s … 2 h), and rolling means of $i^2$, |speed|, |torque| and coolant temperature.
Discretising the RC equation shows the temperature *is* an EWMA of the heating history, and a motor has several thermal
masses (winding: seconds-minutes; iron/housing: minutes-hours), so we give the model several time scales.

**11. Which features are exact physical quantities and which are proxies?**
Exact (given the assumed units): $i^2$, $|i|$, $|u|$, |speed|, |torque|, ambient−coolant, and mechanical power
$T\cdot\omega\cdot 2\pi/60$. Proxies: electrical power $1.5(u_d i_d+u_q i_q)$ (depends on the amplitude- vs
power-invariant d-q convention, which the dataset does not state) and the loss estimate electrical − mechanical power.

**12. Are the units known?**
No; the files contain no data dictionary. Ranges (e.g. speed up to 6000, torque ±260, currents ±300, voltages ±130,
temperatures 9-141) are consistent with rpm, N·m, A, V and °C, and the values are not standardised. That is an
assumption we state openly.

---

## C. Data, validation and leakage

**13. Why not randomly split the samples into train and test?**
Samples are 0.5 s apart and nearly identical to their neighbours, so a random split puts near-duplicates of every
test row into the training set - the model is rewarded for memorising the profile, not for generalising to new ones.
We split by whole `profile_id`: 49 train / 10 validation / 10 test profiles.

**14. How exactly did you split, and is stratifying by peak temperature a leak?**
Profiles are sorted by peak winding temperature, cut into chunks of 7, and each chunk gives one validation and
one test profile by a seeded draw. This only decides *which whole profile goes where* so that mild and severe profiles
appear in every split; no sample-level information crosses splits and the peak is not a model input. The manifest is saved.

**15. What is target leakage? What did you exclude?**
Leakage is information in the features that would not be available in real use (or that is a near-copy of the target).
`stator_tooth`, `stator_yoke` and `pm` are measured temperatures of the same machine (corr with the winding ≈
0.97 / 0.87 / 0.79), so using them would make the problem trivial and defeat the purpose of a
sensor-independent estimator. `profile_id` only tells which recording a row came from. Lagged winding temperature is also
forbidden. Code (`assert_no_forbidden`) and tests enforce this.

**16. Why can a very high R² be misleading?**
R² compares to predicting the mean; with a wide temperature range (here about 19-141 °C) a model that is
9.53 °C off on average can still show R² ≈ 0.795. It also hides profile-to-profile differences and tail errors. We
report MAE in °C, p95, per-profile MAE and the worst profile (3.16 °C for Track B).

**17. How did you prevent scaling/imputation leakage?**
Scalers are inside scikit-learn `Pipeline`s that are fitted on training rows only. The raw data have no missing values, so no
imputation was needed; the only NaNs are the deliberate cold-start values of rolling windows, which are not filled with
arbitrary numbers.

**18. Why group-aware cross-validation?**
For the same reason as the split: folds must hold out whole profiles. Our 5-fold `GroupKFold` gave a mean ± std of
2.62 ± 0.25 °C for Track B and 9.84 ± 1.05 °C for Track A, showing how strongly the
result depends on which profiles are held out.

**19. How do you know the test set was not used for tuning?**
`train.py` only reads train/validation profiles and writes `selection.json` ("test profiles not read"). `evaluate.py`
refuses to run without it, verifies the split manifest and evaluates once. A test checks that training never uses test
profiles and that evaluation requires the frozen selection.

**20. Is your validation score optimistic?**
Slightly: LightGBM's early stopping and the model selection both use the validation profiles. That is why final claims use
the untouched test profiles and why we also report group-CV.

---

## D. Models

**21. How does gradient boosting work, and what is early stopping?**
Start with the mean; repeatedly fit a small tree to the current residuals (negative gradient of the squared loss) and add it
with a small learning rate. Early stopping watches the validation MAE and stops after 100 rounds without
improvement, choosing the number of trees automatically and limiting overfitting.

**22. Why LightGBM instead of deep learning?**
Tabular data with a few dozen features: boosted trees are typically as accurate as neural nets there, train in seconds on a CPU,
are small (857 kB) and fast (3.5 ms), handle NaN natively and are easier to explain. Our history features
already give the tree the memory a recurrent network would learn. Sequence networks are a reasonable future comparison, not a
requirement.

**23. Track A barely beats Ridge regression. Why?**
Because the limiting factor is missing information (thermal history), not model flexibility. Group-CV gives Ridge
10.53 °C vs 9.84 °C for tuned LightGBM on Track A. More flexible functions of the same present-moment
inputs cannot recover what those inputs do not contain.

**24. Did history features really help? How sure are you?**
Yes, by a wide margin: validation MAE 9.42 °C (instantaneous) vs 2.32 °C for the best history set (seed std
≈ 0.08 °C), group-CV 2.62 vs 9.84 °C, and on the test profiles 2.35 vs 9.53 °C. Differences
*among* history sets are smaller and less certain; the mechanism behind the gain from short rolling windows was not isolated.

---

## E. Results and errors

**25. Why is the worst profile worse? What does that tell you?**
Profile 21 has MAE 3.16 °C. It lasts 2.4 h, starts at 20 °C and peaks at 105 °C (coolant mean 19 °C). The model under-estimates the hottest 10 % of the profile by 0.2 °C on average; the MAE is 10.2 °C in the first 10 min and 2.6 °C afterwards. The large early error is the cold-start phase: the 60 s / 300 s rolling windows are NaN at first and the estimate can jump when they become available (see the figure). It shows that the average hides profile-specific failure, so the
worst profile must be reported and operating conditions outside the training experience remain a risk.

**26. How accurate is the estimate in the first minutes, and after a restart?**
Track B error is highest at the start of a profile when the history buffer is empty (5.91 °C MAE in the first 10 min
vs 2.13 °C after 2 h). If the buffer is emptied on an already-warm motor, the 30 min after the restart show MAE
5.38 °C instead of 2.09 °C. A real deployment should initialise state (e.g. from a stored state or a
conservative rule) rather than start blind.

**27. What do permutation importances say, and what can't they say?**
Track B relies mostly on 'load history (i², speed, torque)' and 'coolant history'; Track A on 'ambient / coolant now'. They show what the model uses to predict, not physical causation; correlated features share credit, so
grouped importances are more reliable than single-feature ones. The role of coolant history is partly a property of this test
bench, where the operator sets the coolant temperature.

---

## F. Prediction vs forecasting and monitoring

**28. How is prediction (estimation) different from forecasting?**
Our estimator infers the *present* winding temperature from present and past signals. Forecasting predicts a *future* value
(here 60 s ahead) and cannot use future load. In the extension, persistence ($T(t+h)\approx T(t)$) has MAE
2.66 °C, a sensor-independent forecast 3.24 °C and a sensor-assisted forecast 2.27 °C; using the current
estimator as a forecaster gives 3.85 °C. They are different tasks with different information sets.

**29. Can residual monitoring identify a physical defect?**
No. The residual only says that model and sensor disagree persistently. Causes include sensor faults, cooling changes, an
unseen operating point, model error, or a real motor problem. The dataset has no labelled faults, so we cannot report
detection accuracy.

**30. How was the alarm band chosen, and how reliable is it?**
Band = 99.5th percentile of the 60 s-smoothed |residual| on validation profiles (±10.8 °C), alarm after
120 s of persistence. On healthy test profiles 0.13 % of the time is in alarm and
2 of 10 profiles raise at least one alarm (false alarms / model error); on validation the figure was
0.01 %. So the band does not transfer perfectly - as expected for an empirical band calibrated on 10 profiles.

**31. What does the injected-offset experiment prove?**
Only the sensitivity of the detector to a constant synthetic bias: +2 °C → 2 newly alarming of 10 profiles (median first alarm 16 min); +5 °C → 3 newly alarming of 10 profiles (median first alarm 17 min); +10 °C → 7 newly alarming of 10 profiles (median first alarm 26 min); +15 °C → 8 newly alarming of 10 profiles (median first alarm 4 min). It does not prove real fault detection.

---

## G. Limits, deployment, context

**32. What are the limits of the system?**
One motor on one test bench; trees do not extrapolate (the app warns for out-of-range inputs); history needs a filled buffer;
coolant is operator-controlled on the bench; units and sampling rate are assumed from the documentation; only
10 test profiles; not a certified protection function.

**33. How would you deploy it?**
Compute features with the streaming implementation (`StreamingHistory`, tested equal to the offline one), call the saved
pipeline (≈ 3.5 ms/sample on a CPU, 857 kB), initialise the history buffer sensibly after power-up,
monitor input ranges and residual drift, version model + metadata + config hash, retrain/validate when hardware or operating
strategy changes, and keep a conservative hardware limit independent of the ML estimate.

**34. Why does the title say GE if the data are from Paderborn?**
The GE Power Systems case study is the assignment framing: monitoring and protecting electrical machines. The public
Paderborn data is the only suitable real-measurement dataset with a measured winding-temperature label we could use
reproducibly. It is not GE data and the results make no claim about GE equipment.

---

### Quick numbers to remember

| | Track A (instantaneous) | Track B (history-aware) |
|---|---|---|
| Test MAE / RMSE / R² | 9.53 / 13.34 / 0.795 | 2.35 / 3.37 / 0.987 |
| p95 abs. error | 27.31 | 6.43 |
| Worst test profile MAE | 13.19 | 3.16 |
| Group-CV MAE (mean ± std) | 9.84 ± 1.05 | 2.62 ± 0.25 |
