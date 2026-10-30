# Robust generalisation experiments

The default `scenario` split is retained for backward compatibility. It holds out angles within every distance/resistance pair, so the test set contains operating conditions that are represented in training.

Two stricter experiments are available:

- `distance`: reserve one complete line-fault distance for testing.
- `resistance`: reserve one complete fault-resistance value for testing.

For either condition-holdout experiment, validation is sampled only from the non-held-out conditions. The held-out condition is never used for training or model selection. The scaler is fitted from training windows only. Cache fingerprints include the split strategy and holdout value so incompatible cached partitions are not reused.

## Run a distance holdout

From the repository root:

```powershell
python -m ml.train --split-strategy distance --holdout-value 20
python -m ml.evaluate --checkpoint artifacts/best_model_distance_20.pt
```

This tests 20 km scenarios after training on 150 km and 280 km scenarios. The model checkpoint stores its split settings, and evaluation reads those settings automatically.

Other distance values: `150` or `280`.

## Run a resistance holdout

```powershell
python -m ml.train --split-strategy resistance --holdout-value 50
python -m ml.evaluate --checkpoint artifacts/best_model_resistance_50p0.pt
```

Valid resistance values: `1`, `50`, or `100` ohms.

## Interpretation

Do not force the reported accuracy to a target. Report the measured result on the untouched holdout, including the split strategy, held-out condition, sample count, per-class metrics, and confusion matrix. A result below 100% may reveal a generalisation gap; a result that remains 100% is still a valid result and should be investigated rather than artificially changed.

These are synthetic simulation-data experiments. They do not establish real-world field performance.
