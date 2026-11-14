# GridSentry AI Dashboard

React + TypeScript + Vite dashboard for the GridSentry FastAPI service.

## Requirements

- Node.js 20 or later
- npm
- GridSentry backend running locally or a configured API URL

## Run locally (Windows PowerShell)

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The API defaults to `http://127.0.0.1:8000`; set `VITE_API_URL` in `.env` if the backend is hosted elsewhere.

## Production build

```powershell
npm ci
npm run build
npm run preview
```

## Features

- Backend health and model metadata status
- Explicitly labelled illustrative three-phase waveform visualisation
- Fault taxonomy for all ten supported classes
- JSON waveform input and API-backed predictions
- Per-class probabilities and inference latency
- Responsive desktop, tablet and mobile layout

The dashboard does not fabricate model metrics or prediction results. Metrics are displayed only when the backend accepts the evaluation artifact's protocol and split-integrity metadata. If the API is unavailable, reported values remain unavailable and inference is disabled.

See the [root README](../README.md), [API specification](../docs/api_spec.md), and [validation status](../docs/validation_status.md).
