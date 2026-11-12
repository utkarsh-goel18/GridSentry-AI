# Validation status and reproducibility notes

## Current status

GridSentry AI is a research prototype evaluated on simulated recordings from one 500 kV Chesf transmission-line benchmark. It is not a certified protection relay and has not been validated on operational grid measurements.

The committed historical `artifacts/evaluation_metrics.json` reports 100% accuracy and F1 on 1,230 test windows. Treat this as a historical benchmark result only. It is not an independently reproduced result and is not evidence of external-grid generalisation.

## Split design and audit

The standard split groups each physical scenario by `(distance_km, resistance_ohm, angle_deg)` and assigns that key consistently across Dataset 1 and Dataset 2. This is intended to prevent the same parameterised scenario appearing in more than one partition.

The evaluation pipeline now preserves per-window scenario metadata in the preprocessing cache and reports:
- Train/validation/test scenario counts.
- Scenario-key overlap across partitions.
- Exact duplicate normalized-window hashes across partitions.
- Mean, minimum and maximum maximum-class probability.
- A 15-bin expected calibration error and multiclass Brier score.

The dashboard suppresses legacy evaluation summaries until a newly generated metrics artifact contains the evaluation protocol and reports both scenario and exact-window partitions as disjoint. This prevents the historical 100% figure from being presented as newly audited performance.

An exact-hash audit cannot detect all near-duplicates or shared simulation lineage. A zero-overlap result is necessary, not sufficient, evidence of independence.

## What remains unproven

The standard random scenario split still samples from the same parameter grid, topology and simulation family. It does not establish performance on:
1. An unseen distance value or line length.
2. An unseen fault resistance.
3. An unseen inception angle distribution.
4. A different transmission-line topology or simulator.
5. Field recordings, sensor noise, CT/PT saturation, switching transients or non-fault events.

The existing ten-class API script is an integration smoke test using one representative recording per class. Its 10/10 result must not be reported as model accuracy.

## Required local reproduction

The raw dataset archives are intentionally not committed. After placing both source ZIP files in the repository root, run:

```powershell
python -m pytest -q
python -m ml.train --model 1d_cnn
python -m ml.evaluate --checkpoint artifacts/best_model.pt --output artifacts/evaluation_metrics.json
```

The preprocessing cache fingerprint was advanced when scenario metadata became part of the cache. A stale cache will be rebuilt automatically. Keep the generated metrics, checkpoint, scaler and split summary together; they must come from the same run.

Before making any external-validity claim, add and report separate group-holdout experiments for distance, resistance and inception angle, followed by cross-topology or field-data validation. Do not tune the model against the final test set, and do not change reported scores to target a preferred percentage.

## API guarantees

- The health endpoint reports whether a model is actually loaded.
- Model metadata comes from the loaded checkpoint and configured class mapping.
- Evaluation metrics are shown only when a readable metrics artifact includes the new split-integrity audit and passes both disjointness checks.
- A direct waveform must be a finite, rectangular matrix with six channels and at least 512 samples.
- A direct waveform longer than 512 samples must include an explicit zero-based `fault_index`.
- The sample_id path uses the Dataset 1 fault index configured in `ml/config.py`.

## Direct waveform API

For an already extracted 512-sample window, send a 6-channel matrix under `waveform`. Each channel must contain exactly 512 finite numeric values.

For a longer recording, provide the zero-based sample index of fault inception using `fault_index` and optionally `pre_fault_samples`. For example, Dataset 1 may use fault index 6527 and pre-fault samples 64 when that time reference matches the recording. Do not reuse it blindly for another dataset.
