# GridSentry AI

**Transmission-line fault classification · Research prototype**

GridSentry AI classifies simulated transmission-line faults from three-phase voltage and current waveforms. The repository contains a retained ML benchmark pipeline, evaluation/audit tooling, a FastAPI inference service, and a React + TypeScript dashboard.

> **Research-use notice:** This is a research/demo prototype trained and evaluated using simulated recordings from a 500 kV transmission-line benchmark. It is not a certified protective relay. Do not connect it to operational protection equipment or use its predictions for switching or grid-control decisions. Field measurements, unseen network topologies, hardware-in-the-loop testing, and safety validation are still required.

## Repository layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI application, request schemas, inference service |
| `frontend/` | React + TypeScript + Vite dashboard |
| `ml/` | Dataset ingestion, preprocessing, model, training, evaluation and audit code |
| `scripts/` | Reproducible robustness and inference smoke-test helpers |
| `tests/` | Python unit and API tests |
| `artifacts/` | Tracked model/scaler files and retained historical research outputs |
| `docs/` | API, methodology, dataset audit, architecture and validation documentation |

## Quick start (Windows PowerShell)

### 1. Python environment and backend

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The backend loads the configured checkpoint and scaler from `artifacts/`. If the checkpoint or scaler is missing or incompatible, the API remains available in a degraded state and prediction returns unavailable rather than fabricating a result.

### 2. Dashboard

In a second terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The default API URL is `http://127.0.0.1:8000`. Set `VITE_API_URL` in `frontend/.env` when the API is hosted elsewhere.

### 3. Production frontend build

```powershell
cd frontend
npm ci
npm run build
npm run preview
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service and model readiness |
| GET | `/model-info` | Loaded model metadata and eligible evaluation summary |
| GET | `/samples` | Representative Dataset 1 sample identifiers (requires the local Dataset 1 archive) |
| POST | `/predict` | Classify a sample identifier or six-channel waveform |

Interactive Swagger documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) while the backend is running. See [API specification](docs/api_spec.md).

## Dataset setup

Raw source archives are intentionally excluded from Git. Place the original archives in the repository root with these exact names:

- `signals (dataset 1).zip`
- `signals_with_compensation (dataset 2).zip`

Do not modify or regenerate source data without recording provenance and checksums. See [dataset audit](docs/dataset_audit.md) and [methodology](docs/methodology.md).

## Results and validation

Read [the evaluation and readiness report](docs/final_evaluation_report.md) and [validation status](docs/validation_status.md) before interpreting any metric.

The retained `artifacts/evaluation_metrics.json` and baseline outputs are historical benchmark artifacts. They predate the current split-integrity metadata contract and are deliberately not exposed by the dashboard as validated performance. The evaluation report records separate random, distance, resistance and angle holdout results; the distance and resistance holdouts show material generalisation weaknesses. These synthetic-benchmark results are not field accuracy.

Run the full local Python test suite:

```powershell
python -m pytest -q
```

Run the four-protocol robustness suite (requires the source dataset archives):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_robustness_suite.ps1
```

The robustness suite writes strategy-specific checkpoints, configs and metrics and leaves the default inference checkpoint untouched. Review the generated outputs before updating any published result.

## Safety and scope

- The model predicts one of ten configured fault classes; it is not an open-set or non-fault detector.
- Neither source archive contains a separately labelled normal or overload class.
- A high softmax score is not proof of calibration, reliability or physical correctness.
- Do not use the prototype for live grid control, relay tripping or operational decisions.

## Development checks

CI runs the Python test suite and a clean frontend production build on pull requests to `main`. Frontend dependencies are installed from the committed lockfile with `npm ci`.
