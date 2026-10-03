# Transportation safety decision-fusion study

This repository contains only **code**, **input**, and **output**. It archives an executed scalar-stream simulation and a participant-disjoint secondary analysis of public driving-simulator event logs. Manuscripts, submission packaging, and document-generation scripts are not included.

## Evidence boundary

The implemented decision layer combines driver-state score E, external-hazard score H, vehicle-telemetry score T, uncertainty descriptors, availability, causal memory, and interactions. A matched two-hidden-layer neural fusion model is evaluated. Facial-emotion recognition, camera-based vehicle detection, tracking, and vehicle control are architectural interfaces, not trained or validated perception modules in this repository. Simulated risk-state labels are not real crash observations. No experiment establishes that accidents were prevented.

## Reproduce

The recorded run used Python 3.14.7 on an Apple arm64 desktop. Exact package versions are pinned in requirements.txt and recorded in output/synthetic/run_manifest.json. Inputs are already supplied, so no dataset login is needed.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python code/simulation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python code/eventstudy.py
python code/test_invariants.py -v
python code/plot_results.py
```

The experiment regenerates input/synthetic and output/synthetic. The event study regenerates derived event-instance CSVs and output/events. Preserve the published results before exploratory changes. Numerical metrics are reproducible with the pinned software; timings and file serialization metadata can differ across machines.

## Synthetic protocol

Each seed produces 80 independent sequences at nominal 10 Hz. Fifty frames are generated; the first forty frames are used as predictors. The same current observations support two distinct targets: current state at offset zero and the stochastic state exactly ten frames later. Future observations and labels never enter feature construction.

| Partition | Seeds | Predictor frames |
|---|---|---:|
| Fit | 6001-6020 | 64,000 |
| Probability calibration | 7001-7005 | 16,000 |
| Threshold selection | 7006-7010 | 16,000 |
| Evaluation | 8001-8030 | 96,000 per scenario |

Latent logits are initialized as independent standard normals and evolved with autoregressive coefficients 0.95, 0.90, and 0.92. Hazard and telemetry innovations have correlation 0.5. Sigmoid-transformed latent states determine a stochastic Bernoulli risk label through a declared interaction equation. Observations contain quality-dependent Gaussian noise, four-percent high-uncertainty outliers, and two-percent natural missingness. The declared generator is an assumption under test, not a vehicle dynamics model.

Seventeen configurations are fitted for each target. The eleven primary baselines include full-feature histogram gradient boosting and a neural model with 32 and 16 rectified hidden units. Five ablations remove quality, temporal memory, interactions, driver evidence, or telemetry. Probability calibration and threshold selection use separate seeds. Thresholds maximize F1 subject to threshold-selection recall at least 0.75, and are frozen in stress tests.

Eight scenarios preserve the same latent labels and paired seeds: clean, additional independent dropout at 10/20/30 percent, a contiguous 15-frame outage per modality, increased observation noise, overconfident quality descriptors, and driver-stream shuffling across sequences. The contiguous outage occupies 30 percent of the generated 50-frame record; overlap with the forty predictor frames varies with its start time. Features hold their previous state during missingness, which makes stale memory a declared failure mechanism.

Primary inference compares contemporaneous clean average precision across thirty seeds. Two-sided Wilcoxon tests and Holm correction cover the eleven baseline comparisons. Bootstrap confidence intervals resample seeds. Ablations, shifted conditions and future-state results are descriptive; they are not additional confirmatory tests.

## Public event-log study

The source is *Driving with Autonomous Aids*, NEMAR on004657 v1.0.0, derived from OpenNeuro ds004657 v1.0.3. Its license is CC0. Source metadata, the event dictionary, event TSV files, and SHA-256 records are included in input. EEG/EOG/ECG/EDA samples, participant demographics, and video are excluded.

A prediction is made at a pedestrian-onset marker. Features use only events strictly earlier than that onset, plus the type of the pedestrian appearing at that instant and the session's predefined assistance condition. The target is **any direct pedestrian-collision marker within the following eight seconds**, not an assertion that the current pedestrian caused that collision. Four- and twelve-second windows are sensitivity analyses. Several onset windows can refer to the same collision; positive onset windows are not unique accidents. Late onsets without a complete future window are censored.

Six outer participant folds keep every participant's sessions together. Within each outer training pool, separate participant groups fit the estimator, calibrate the probability, and choose a threshold targeting five-percent false-positive rate on threshold-selection negatives. No calibration, threshold or fit participant enters that fold's test group. The exact groups and split seeds are retained. Cluster bootstrap intervals resample participants.

The event study evaluates contextual and interaction fusion, not the synthetic UATIF quality gate and not raw-image emotion or vehicle perception. Its warning lead time is measured from each positive onset to the earliest collision within its outcome window; an eight-second window does not imply eight seconds of advance warning.

## Files

- code/fusion.py: causal features, calibrated classifiers and portable UATIF inference.
- code/simulation.py: input generation, fitted models, seed results, paired tests and timing.
- code/eventstudy.py: prospective labels, historical features and participant-disjoint evaluation.
- code/test_invariants.py: reproducibility, causality, censoring, split and streaming checks.
- code/plot_results.py: five figures reconstructed from outputs.
- input/config.json: frozen numerical protocol and seeds.
- input/synthetic/*.npz: the actual generated arrays, including latent references and targets.
- input/events/: public CC0 event inputs and original dataset metadata.
- input/event_instances_h*.csv: derived prospective analysis records.
- input/source_provenance.json: dataset attribution and event-input hashes.
- output/synthetic/: seed metrics, summaries, paired contrasts, fitted models, full clean predictions, portable inference parameters and runtime scope.
- output/events/: every held-out prediction, participant partitions, cluster bootstrap samples and metric tables.
- output/figures/: PNG, PDF and SVG figures.
- output/research_integrity.json: final file hashes and recorded validation.

## Interpretation

A nonlinear or neural baseline can match or exceed the proposed interpretable layer. Missingness-aware temporal features account for much of the advantage over legacy zero-imputed fusion. Quality bias and extra noise degrade calibration. The empirical interaction contrast is uncertain, precision is low at the chosen alert budget, and the experiment remains a simulator-event analysis. The complete outputs retain these findings.

Availability status is separate from a numeric probability. An all-missing input is marked unavailable; holding a temporal state is not evidence that an operational vehicle is safe. The code does not actuate brakes or steering.

## Dataset citation

Metcalfe, J., Marathe, A., Johnson, T., Gordon, S., Touryan, J., and King, K. (2026). Driving with Autonomous Aids, version 1.0.0. NEMAR. https://doi.org/10.82901/nemar.on004657. Original source: https://doi.org/10.18112/openneuro.ds004657.v1.0.3.
