# GridSentry AI — API Specification

The GridSentry AI REST backend is implemented in FastAPI and exposes endpoints for health monitoring, model metadata retrieval, dataset sample exploration, and real-time transmission-line fault classification.

- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Documentation**:
  - Swagger UI: `http://127.0.0.1:8000/docs`
  - ReDoc: `http://127.0.0.1:8000/redoc`

---

## 1. System Health

### `GET /health`
Returns the operational status of the service, whether the PyTorch model is loaded in memory, and the active compute device (`cpu` or `cuda`).

#### Response: `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2026-10-26T06:33:44.923163Z"
}
```

---

## 2. Model Metadata & Taxonomy

### `GET /model-info`
Provides model architecture specifications, channel order, input dimensions, class taxonomy, and held-out test evaluation metrics.

#### Response: `200 OK`
```json
{
  "model_name": "1d_cnn",
  "model_architecture": "GridSentry1DCNN (4 Conv1D blocks, BatchNorm, MaxPool, AdaptiveAvgPool, Linear)",
  "parameters_count": 111306,
  "num_classes": 10,
  "classes": [
    "AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"
  ],
  "input_spec": {
    "num_channels": 6,
    "channels": ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"],
    "window_length": 512,
    "sampling_rate_hz": 15360,
    "nominal_freq_hz": 60,
    "samples_per_cycle": 256,
    "duration_ms": 33.33
  },
  "evaluation_summary": {
    "test_accuracy": 1.0,
    "macro_f1": 1.0,
    "weighted_f1": 1.0,
    "best_epoch": 10,
    "val_loss": 0.00003
  },
  "dataset_summary": {
    "total_physical_scenarios": 171,
    "train_scenarios_count": 117,
    "val_scenarios_count": 27,
    "test_scenarios_count": 27,
    "train_windows": 15960,
    "val_windows": 1230,
    "test_windows": 1230
  }
}
```

---

## 3. Dataset Samples Catalog

### `GET /samples`
Returns a curated list of representative dataset recordings from the local archive, suitable for populating a frontend dropdown menu or running manual test queries.

#### Response: `200 OK`
```json
{
  "total_samples": 10,
  "samples": [
    {
      "sample_id": "sim_Chesf_500_Classifier_LOC3_AT_150_10_100.000.mat",
      "fault_class": "AG",
      "description": "Single Line-to-Ground Fault (Phase A)",
      "distance_km": 150,
      "resistance_ohm": 100.0,
      "angle_deg": 10,
      "compensation": 0.0
    },
    {
      "sample_id": "sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat",
      "fault_class": "AB",
      "description": "Line-to-Line Fault (Phases A and B)",
      "distance_km": 20,
      "resistance_ohm": 1.0,
      "angle_deg": 0,
      "compensation": 0.0
    }
  ]
}
```

---

## 4. Fault Classification Inference

### `POST /predict`
Classifies transmission-line faults. Accepts either:
1. `sample_id`: Name of a local dataset `.mat` file (e.g. `'sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat'`).
2. `waveform`: A 2D numerical array of shape `[6, N]` or `[N, 6]` with $N \ge 512$ representing channels `[Va, Vb, Vc, Ia, Ib, Ic]`.

> **Note**: Exactly one of `sample_id` or `waveform` must be provided. Path traversal characters (`..`, `/`, `\`) in `sample_id` are strictly forbidden.

#### Option A: Prediction by Dataset Sample Identifier

**Request Payload**:
```json
{
  "sample_id": "sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"
}
```

**Response: `200 OK`**:
```json
{
  "predicted_fault": "AB",
  "fault_description": "Line-to-Line Fault (Phases A and B)",
  "confidence": 1.0,
  "class_probabilities": {
    "AG": 0.0,
    "BG": 0.0,
    "CG": 0.0,
    "AB": 1.0,
    "BC": 0.0,
    "CA": 0.0,
    "ABG": 0.0,
    "BCG": 0.0,
    "CAG": 0.0,
    "ABC": 0.0
  },
  "input_type": "sample_id",
  "sample_metadata": {
    "fault_label": "AB",
    "sampling_rate": 15360,
    "fault_time": 0.42493,
    "raw_fault": "AB",
    "standard_fault": "AB",
    "distance_km": 20,
    "angle_deg": 0,
    "resistance_ohm": 1.0,
    "compensation": 0.0,
    "dataset": "Dataset 1 (Uncompensated)",
    "source_file": "sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"
  },
  "inference_time_ms": 2.15
}
```

#### Option B: Prediction by Direct Waveform

**Request Payload**:
```json
{
  "waveform": [
    [ -104379.5, -68201.9, ... ],  // Channel 0: Va (512 points)
    [ -455422.1, -444071.6, ... ], // Channel 1: Vb (512 points)
    [  559801.6,  512273.6, ... ], // Channel 2: Vc (512 points)
    [       0.1,      87.4, ... ], // Channel 3: Ia (512 points)
    [       0.4,      83.2, ... ], // Channel 4: Ib (512 points)
    [      -0.5,    -170.7, ... ]  // Channel 5: Ic (512 points)
  ]
}
```

**Response: `200 OK`**:
```json
{
  "predicted_fault": "CA",
  "fault_description": "Line-to-Line Fault (Phases C and A)",
  "confidence": 0.9999,
  "class_probabilities": {
    "AG": 0.000001,
    "BG": 0.0,
    "CG": 0.0,
    "AB": 0.0,
    "BC": 0.0,
    "CA": 0.99992,
    "ABG": 0.0,
    "BCG": 0.0,
    "CAG": 0.000079,
    "ABC": 0.0
  },
  "input_type": "direct_waveform",
  "sample_metadata": null,
  "inference_time_ms": 1.84
}
```

---

## 5. Error Responses

| Status Code | Scenario | Response Body |
|---|---|---|
| `400 Bad Request` | Neither `sample_id` nor `waveform` provided | `{"detail": "Either 'sample_id' or 'waveform' must be provided."}` |
| `400 Bad Request` | Both `sample_id` and `waveform` provided | `{"detail": "Provide either 'sample_id' or 'waveform', not both."}` |
| `400 Bad Request` | Path traversal attempt in `sample_id` | `{"detail": "sample_id must be a clean filename/identifier without path separators."}` |
| `400 Bad Request` | Waveform has wrong channel count | `{"detail": "Expected 6 channels, but got 4 channels."}` |
| `400 Bad Request` | Waveform length $< 512$ samples | `{"detail": "Signal has 200 time steps; minimum 512 required."}` |
| `404 Not Found` | Non-existent recording in archive | `{"detail": "Sample recording not found: Recording missing.mat not found in signals (dataset 1).zip"}` |
| `503 Service Unavailable` | Model checkpoint not yet loaded | `{"detail": "Model is not ready. Please verify artifacts/best_model.pt."}` |
