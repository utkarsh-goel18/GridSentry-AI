# Validation status and reproducibility notes

## API guarantees

- The health endpoint reports whether a model is actually loaded.
- Model metadata comes from the loaded checkpoint and configured class mapping.
- Evaluation metrics are returned only when a readable metrics artifact contains the required values. Missing or malformed metrics are represented as null rather than replaced with perfect scores.
- A direct waveform must be a finite, rectangular matrix with six channels and at least 512 samples.
- A direct waveform longer than 512 samples must include an explicit zero-based fault_index. The API no longer guesses a fault location from recording length.
- The sample_id path uses the Dataset 1 fault index configured in ml/config.py.

## Evaluation caveat

The committed artifacts/evaluation_metrics.json currently reports 100% accuracy and F1 on 1,230 test windows. This is a recorded artifact, not an independently reproduced result in this repository session.

The scenario split assigns the same (distance_km, resistance_ohm, angle_deg) key to the same partition across Dataset 1 and Dataset 2. This is a useful leakage control, but it does not by itself establish generalization to another transmission-line topology, simulation model, or real-world measurements.

Before presenting the 100% figure as independently verified:
1. Rebuild the preprocessing cache from the raw archives.
2. Re-run training and evaluation from a clean environment.
3. Verify Dataset 1 / Dataset 2 MATLAB axis ordering and channel orientation against source metadata.
4. Inspect duplicate and near-duplicate signals across partitions.
5. Preserve the newly generated metrics and confusion matrix with the exact checkpoint used.

## Direct waveform API

For an already extracted 512-sample window, send a 6-channel matrix under waveform. Each channel must contain exactly 512 finite numeric values.

For a longer recording, provide the zero-based sample index of fault inception using fault_index and optionally pre_fault_samples. For example, a Dataset 1 recording may use fault_index 6527 and pre_fault_samples 64 if that index matches its time reference. Do not reuse it blindly for a different dataset.
