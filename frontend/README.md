# GridSentry AI Dashboard

React + TypeScript + Vite dashboard for the GridSentry FastAPI service.

## Run locally

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

The dashboard expects the API at `http://127.0.0.1:8000` by default. Set `VITE_API_URL` in `.env` when the API is hosted elsewhere.

## Production build

```powershell
npm run build
npm run preview
```

## Features

- Backend health and model metadata status
- Explicitly labelled illustrative three-phase waveform visualisation
- Fault taxonomy for all ten supported classes
- JSON waveform upload/paste and live API prediction
- Per-class probabilities and inference latency
- Responsive layout for desktop, tablet and mobile

The dashboard does not fabricate model metrics or prediction results. If the API is unavailable, reported values remain unavailable and inference is disabled.
