# GridSentry AI — Evaluation and Readiness Report

## Executive summary

GridSentry AI is a 10-class, six-channel 1D-CNN classifier for simulated transmission-line fault waveforms. It accepts three-phase voltage and current channels (Va, Vb, Vc, Ia, Ib, Ic) and predicts one of ten fault labels.

The results show strong performance on the random and angle-holdout partitions, but materially weaker generalization to unseen distance and fault-resistance settings. The model is therefore a promising benchmark prototype, not a field-validated protection system.

## Evaluation results

| Evaluation split | Test samples | Accuracy | Macro-F1 | Temperature |
|---|---:|---:|---:|---:|
| Random scenario split | 1,230 | 100.00% | 100.00% | 1.0000 |
| Distance holdout | 2,470 | 62.11% | 59.81% | 1.0079 |
| Resistance holdout | 2,470 | 66.07% | 64.81% | 1.3020 |
| Angle holdout | 1,220 | 100.00% | 100.00% | 1.0000 |

All reported scenario-grouped evaluations recorded disjoint train/validation/test scenario sets and zero exact duplicate windows across partitions. The grouping key is (distance_km, resistance_ohm, angle_deg).

These are separate evaluation protocols and should not be interpreted as one pooled test score. The random and angle-holdout results do not cancel out the weaker distance and resistance results.

## Generalization findings

### Distance holdout

Accuracy is 62.11% and macro-F1 is 59.81%. Recall is especially weak for:
- BCG: 7.69%
- CAG: 6.07%
- ABG: 43.32%

BC and CA recall are 100%, but their precision is low (35.09% and 37.77%), indicating substantial false positives for those classes.

### Resistance holdout

Accuracy is 66.07% and macro-F1 is 64.81%. Recall is especially weak for:
- BCG: 17.81%
- CAG: 30.77%
- ABG: 58.30%

The model's generalization to unseen resistance values is better than the distance-holdout result, but still not adequate to claim robust performance across operating conditions.

### Angle holdout

The angle-holdout test reached 100% accuracy and macro-F1 on 1,220 samples. This is evidence for performance on the particular simulated angle split used here, not proof of generalization to arbitrary grid conditions or real measurements.

## Probability calibration

Temperature scaling is fitted on validation logits and labels only. The held-out test partition is not used to fit the temperature. A positive scalar temperature preserves the predicted class, so calibration changes confidence scores but does not change accuracy.

| Split | Temperature | Raw mean max probability | Calibrated mean max probability |
|---|---:|---:|---:|
| Random | 1.0000 | 99.9998% | 99.9998% |
| Distance holdout | 1.0079 | 75.2306% | 75.0177% |
| Resistance holdout | 1.3020 | 74.6564% | 69.7001% |
| Angle holdout | 1.0000 | 99.9983% | 99.9983% |

A temperature close to 1 means the validation set did not support a substantial confidence adjustment. Calibration is not intended to artificially reduce confidence or accuracy.

## Model scope

- Architecture: GridSentry1DCNN (1D CNN)
- Trainable parameters: 111,306
- Input: 6 waveform channels, 512 samples per window
- Sampling rate: 15,360 Hz
- Output classes: AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC

The present label set covers phase-to-ground, phase-to-phase, double-phase-to-ground, and three-phase faults. **Overload is not a class in the current dataset/model**, so the current implementation does not detect overload despite it appearing in the broader problem statement.

## Important limitations

1. The benchmark is simulated (500 kV Chesf dataset); no independent field measurements or external grid topology validation have been reported.
2. Perfect observed accuracy on a finite synthetic test split is possible, but it does not establish perfect physical performance or operational reliability.
3. Distance and resistance holdout results expose meaningful generalization weaknesses, especially for BCG and CAG.
4. The model predicts among the ten configured classes; it is not an open-set detector and should not be assumed to safely reject unknown conditions.
5. This work is a research/demo prototype. It must not be used as a protection relay or for live grid control.

## Recommended next steps

1. Prioritize additional training/evaluation data at unseen distances and resistance values, with particular attention to BCG, CAG, and ABG.
2. Add an explicit overload class only after obtaining correctly labelled overload waveforms and defining a leakage-safe scenario split.
3. Evaluate on a genuinely independent simulation configuration or measured dataset, not only alternate partitions of the same benchmark.
4. Add an unknown/out-of-distribution rejection strategy and test its false-alarm and missed-detection trade-offs.
5. Report confusion matrices, per-class recall, ECE/Brier score, and the exact split protocol alongside any headline accuracy.

## Reproducibility

Example evaluation commands (run from the repository root after placing the corresponding checkpoint and dataset archives in `artifacts/` and the configured dataset paths):

```powershell
python -m ml.evaluate --checkpoint artifacts/best_model.pt --output artifacts/evaluation_metrics.json --split-strategy random
python -m ml.evaluate --checkpoint artifacts/best_model_distance_audit.pt --output artifacts/evaluation_metrics_distance_audit.json --split-strategy distance_holdout --no-write-shared-artifacts
python -m ml.evaluate --checkpoint artifacts/best_model_resistance_audit.pt --output artifacts/evaluation_metrics_resistance_audit.json --split-strategy resistance_holdout --no-write-shared-artifacts
python -m ml.evaluate --checkpoint artifacts/best_model_angle_audit.pt --output artifacts/evaluation_metrics_angle_audit.json --split-strategy angle_holdout --no-write-shared-artifacts
```

Each evaluation writes a checkpoint-specific calibration sidecar. The inference API verifies the sidecar's checkpoint SHA-256 and validation-only provenance before applying it; otherwise it falls back to raw softmax and marks confidence as uncalibrated.
