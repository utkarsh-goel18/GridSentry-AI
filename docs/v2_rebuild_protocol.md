# GridSentry AI V2 — Rebuild and Acceptance Protocol

## Non-negotiable rules

- Preserve existing history and artifacts; V2 work happens on `rebuild/v2`.
- Never alter metrics to make them look more plausible.
- Never report training/validation results as test results.
- Never describe synthetic-benchmark performance as field accuracy.
- Never claim Dataset 1 and Dataset 2 are independent without evidence.
- A failed or missing model must produce an unavailable/degraded state, not a fabricated prediction.

## Phase 1 — Repository and data inventory

- [ ] Record Git commit, branch and working-tree state.
- [ ] Record archive file names, byte sizes and SHA-256.
- [ ] Inspect archive members and confirm expected MAT/HDF5 formats.
- [ ] Validate all class labels and metadata parsing.
- [ ] Validate waveform dimensions, channel order, sampling rate, units and finite values.
- [ ] Confirm fault inception index and window bounds independently for each source.
- [ ] Align fault-relative waveform segments before any correlation or duplicate analysis.
- [ ] Compare metadata and signal fingerprints to identify exact/near duplicates.
- [ ] Document simulation lineage and any shared source model.

## Phase 2 — Canonical dataset layer

Each recording must map to a canonical record containing source ID, source archive hash, source file, class label, distance, resistance, angle, compensation, sampling rate, fault index/time, channel names, physical units and signal array.

Source adapters must reject malformed data with actionable errors. Do not infer missing metadata silently. Keep Dataset 1 and Dataset 2 readers separate, then unify only after validation.

## Phase 3 — Leakage-resistant evaluation

- [ ] Define a physical scenario/group key from simulation provenance.
- [ ] Group all related recordings and derived windows before splitting.
- [ ] Ensure no group appears in more than one split.
- [ ] Fit scaling parameters on training groups only.
- [ ] Keep validation for model selection and test for one-time final evaluation.
- [ ] Save a machine-readable split manifest and dataset fingerprint.
- [ ] Include per-class precision, recall, F1, support, confusion matrix and confidence calibration.
- [ ] Add cross-dataset transfer only after source independence and signal compatibility have been assessed.

## Phase 4 — Model comparison

Minimum comparisons:
- physics-informed threshold/rule baseline where defensible;
- classical feature baseline (e.g. logistic regression or random forest);
- current 1D CNN benchmark;
- one alternative temporal architecture only if it adds value.

Evaluate sensitivity to noise, timing jitter, scaling drift, fault resistance, fault distance, inception angle and compensation. Report all evaluation conditions.

## Phase 5 — Application acceptance

- [ ] Backend health reports degraded if checkpoint/scaler are unavailable.
- [ ] Invalid waveform shapes and non-finite values return client errors.
- [ ] Frontend builds cleanly and is responsive.
- [ ] Frontend does not invent metrics or predictions when API is offline.
- [ ] Prediction request and response match API schemas.
- [ ] CORS is configurable and restricted in deployment.
- [ ] API and frontend integration tests pass.
- [ ] Deployment secrets and dataset archives are not committed.

## Release checklist

A release is complete only when code review, automated tests, reproducible model evaluation, API smoke test, frontend production build, deployment configuration and limitations documentation are all complete. A successful frontend build alone is not a completed ML system.
