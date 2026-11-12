# GridSentry AI V2 — Architecture

## Product boundaries

GridSentry is split into a browser client, API boundary, inference service, and offline model-development pipeline. Training code must not be imported by the browser and the API must not silently retrain or mutate model artifacts.

```text
React + TypeScript dashboard
        │ HTTPS / JSON
        ▼
FastAPI routes + Pydantic validation
        │
        ▼
Inference service ── checkpoint + training scaler
        │
        └── response: class, probabilities, latency, metadata

Offline research pipeline
  source archives → ingestion → validation → provenance manifest
  → group-aware split → train-only preprocessing → baselines/models
  → evaluation report + versioned model artifact
```

## Frontend

- React 18, TypeScript and Vite.
- Recharts for signal visualisation; Lucide icons.
- API URL configured through `VITE_API_URL`.
- No hard-coded prediction or test metric fallback.
- Illustrative waveform chart is explicitly labelled as not live telemetry.
- Prediction view accepts JSON waveform data and displays the backend response only.

## Backend

FastAPI owns request validation, health reporting, model metadata and inference orchestration. Model loading must fail closed: an unavailable checkpoint yields a degraded health response and disabled inference, not a fake success state.

CORS must be restricted to explicitly configured deployment origins. The default local development origin is localhost:5173.

## ML and data

The existing `ml/` pipeline remains available as a historical baseline during V2 development. A replacement pipeline must separate:
1. Archive readers and source-specific format adapters.
2. Signal validation and canonical channel representation.
3. Scenario identity, source provenance and duplicate detection.
4. Group-level split assignment.
5. Train-only scaling and window extraction.
6. Model training and evaluation.

No result is called independent generalisation until the test groups and data lineage support that claim.

## Artifact contract

Every model release should carry:
- architecture and code revision;
- source archive SHA-256 hashes;
- class ordering and channel order;
- sampling rate, units and window/fault alignment;
- split manifest and random seed;
- training-only scaler;
- metrics with per-class support;
- calibration/uncertainty analysis;
- known limitations.

## Deployment

The frontend can be hosted as static assets. The API requires a Python runtime, model artifacts and access to any sample archive it serves. Do not expose dataset paths or arbitrary local file access to public requests. Use HTTPS, explicit CORS origins, request size limits and authentication/rate limiting before public deployment.
