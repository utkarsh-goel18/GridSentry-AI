# GridSentry AI

**Transmission-line fault intelligence · Research prototype**

GridSentry AI is a research prototype for classifying transmission-line faults from three-phase voltage and current waveforms. The repository contains the legacy ML benchmark and a V2 application track with a React dashboard and FastAPI inference service.

> **Research-use notice:** Current datasets are simulated recordings for a 500 kV transmission-line benchmark. The project is not a certified protective relay and must not be connected to operational grid protection or used to make switching decisions. Real-world, unseen-topology and hardware-in-the-loop validation are required.

## V2 dashboard

The React + TypeScript dashboard lives in `frontend/`. It displays API health and model metadata, explains the supported fault taxonomy, and sends user-provided waveform JSON to the inference API. It deliberately labels the waveform chart as illustrative and never invents model metrics or predictions.

### Start the backend

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API expects the configured model checkpoint and scaler under `artifacts/`. If they are missing or incompatible, the service reports a degraded state and prediction remains unavailable.

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open http://localhost:5173. Set `VITE_API_URL` in `frontend/.env` if the API is hosted elsewhere.

### Production frontend build

```powershell
cd frontend
npm run build
npm run preview
```

## API

- `GET /health` — service and model readiness
- `GET /model-info` — model metadata and available evaluation summary
- `GET /samples` — representative Dataset 1 sample identifiers
- `POST /predict` — classify a sample identifier or six-channel waveform

Interactive API documentation is available at `/docs` when the backend is running.

## Dataset inputs

The raw ZIP archives are not committed to Git. Place the original archives in the repository root using the configured names:

- `signals (dataset 1).zip`
- `signals_with_compensation (dataset 2).zip`

Do not rename, modify or regenerate source data without recording provenance and checksums. See [dataset audit](docs/dataset_audit.md) and [V2 rebuild protocol](docs/v2_rebuild_protocol.md).

## Existing ML benchmark

The original 1D CNN and baseline MLP, dataset loader, training/evaluation scripts and historical artifacts are retained for traceability. Previously reported 100% scores are historical benchmark outputs; they are not a V2 result or evidence of field reliability. See [validation status](docs/validation_status.md).

## Development

```powershell
python -m pytest -v
```

Frontend checks:

```powershell
cd frontend
npm install
npm run build
```

See [V2 architecture](docs/v2_architecture.md) for the application boundaries and [V2 rebuild protocol](docs/v2_rebuild_protocol.md) for the acceptance gates.
