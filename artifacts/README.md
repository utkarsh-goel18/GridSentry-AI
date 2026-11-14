# Tracked artifact inventory

This directory contains a small set of model/runtime files and retained research outputs. It does **not** contain the raw source dataset archives; those are kept locally and excluded from Git.

## Runtime files

| File | Role |
|---|---|
| `best_model.pt` | Default checkpoint loaded by the inference service |
| `scaler_params.json` | Six-channel normalization parameters used by inference |
| `class_mapping.json` | Configured fault-class mapping |

The checkpoint and scaler must belong to the same training run. Do not replace one without regenerating and validating the matching pair.

## Retained historical benchmark files

| File | Role and interpretation |
|---|---|
| `baseline_model.pt` | Retained baseline MLP checkpoint |
| `baseline_evaluation_metrics.json` | Historical baseline evaluation output |
| `evaluation_metrics.json` | Historical default-model evaluation output |
| `config.json` | Legacy baseline run configuration, not the current inference configuration |
| `dataset_split_summary.json` | Retained split summary |
| `dataset_audit_report.json` | Retained dataset audit output |

The legacy metric JSON files predate the current evaluation-protocol and split-integrity metadata contract. The backend therefore does not expose them as validated dashboard metrics. Consult [the evaluation report](../docs/final_evaluation_report.md) and [validation status](../docs/validation_status.md) for the documented robustness results and caveats.

## Figures

`figures/` contains training-history and waveform-analysis figures used as research records. The duplicate baseline confusion-matrix image was removed because it had the exact same Git blob content as the 1D-CNN confusion-matrix image; keep one canonical copy rather than two identically named results.

## Maintenance rules

- Do not commit raw dataset archives, extracted recordings, preprocessing caches, virtual environments, local secrets or build output.
- Keep deployable checkpoints only when their matching scaler, class ordering, provenance and evaluation record are available.
- Do not overwrite historical metrics with a different experiment. Use strategy-specific filenames for new runs.
- Remove or replace an artifact only after checking code, documentation and dashboard references.
