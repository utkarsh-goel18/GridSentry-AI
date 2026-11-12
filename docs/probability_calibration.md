# Probability calibration

GridSentry reports classification accuracy and confidence as different quantities.
Accuracy is the fraction of evaluated labels predicted correctly. The maximum
softmax output is a model score; by itself it is not a trustworthy probability.

## Calibration method

The evaluation pipeline fits one positive scalar temperature using validation
logits and validation labels only. The fitted temperature is applied to logits
before softmax at inference:

```
calibrated_probabilities = softmax(logits / temperature)
```

The held-out test partition is not used to fit or select the temperature.
Temperature scaling preserves the argmax class for a positive temperature, so
it does not reduce accuracy or manufacture errors. It can increase or decrease
confidence depending on validation calibration.

## Reproduce calibration for the production checkpoint

From the project root, with the source dataset archives available:

```powershell
python -m ml.evaluate --checkpoint artifacts/best_model.pt --output artifacts/evaluation_metrics.json --split-strategy random
```

This writes:
- `artifacts/calibration.json` — temperature, checkpoint SHA-256, split strategy,
  validation sample count and provenance flags.
- `artifacts/evaluation_metrics.json` — raw and calibrated test confidence
  diagnostics, alongside unchanged accuracy/F1 metrics.

The API loads calibration only when the SHA-256 matches its currently loaded
checkpoint and the artifact declares validation-only fitting. If no matching
artifact exists, inference falls back to raw softmax and explicitly marks the
confidence as uncalibrated. It never invents a temperature or silently applies
a calibration fitted for a different checkpoint.

## Interpretation and limitations

Calibration is not a mechanism to make confidence look lower. If validation
data supports high confidence, a fitted temperature may remain near 1 or even
increase confidence. Review validation NLL before and after calibration, test
ECE and Brier score, reliability diagrams, class-wise behavior and performance
on genuinely independent operating conditions.

A finite synthetic test split can in principle yield 100% observed accuracy.
That does not prove perfect physical performance or field reliability. Keep
the exact benchmark scope and test sample count beside any reported score.
Independent topology and field measurements are still required.
