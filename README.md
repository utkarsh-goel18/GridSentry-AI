# GridSentry-AI

GridSentry AI detects and classifies transmission-line faults from six-channel voltage and current waveform windows.

## Model evaluation

The default evaluation uses a scenario-level holdout. For stricter tests on operating conditions absent from training, see [Robust generalisation experiments](docs/robust_generalization.md). These experiments support full-distance and full-resistance holdouts without changing the existing benchmark.
