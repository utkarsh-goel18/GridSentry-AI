"""
GridSentry AI — FastAPI application.
"""
import datetime
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse, ModelInfoResponse, InputSpec, EvaluationSummary,
    PredictRequest, PredictResponse, SampleListResponse, SampleIdentifier
)
from backend.inference import engine
from ml.config import (
    STANDARD_CLASSES, WINDOW_LENGTH, SAMPLING_RATE_HZ,
    NOMINAL_FREQ_HZ, SAMPLES_PER_CYCLE, EVALUATION_METRICS_PATH,
    ARTIFACTS_DIR, CHANNELS
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine.load()
        print("GridSentry AI model and scaler loaded.")
    except Exception as exc:
        # Keep the API process available so /health can report degraded state.
        engine.model = None
        print(f"GridSentry AI startup: model unavailable: {exc}")
    yield


app = FastAPI(
    title="GridSentry AI — Transmission Line Fault Classification API",
    description="1D CNN classification of six-channel three-phase voltage/current waveforms.",
    version="1.1.0",
    lifespan=lifespan
)

# Restrict origins through GRID_SENTY_CORS_ORIGINS in deployment if the UI is hosted elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    ready = engine.model is not None
    return HealthResponse(
        status="healthy" if ready else "degraded",
        model_loaded=ready,
        device=str(engine.device),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


def _read_evaluation_summary():
    """Return metrics only when the artifact contains all required finite values."""
    if not EVALUATION_METRICS_PATH.exists():
        return None
    try:
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        required = ("test_accuracy", "macro_f1", "weighted_f1", "best_epoch", "val_loss")
        if any(key not in data or data[key] is None for key in required):
            return None
        values = [float(data[k]) for k in required if k != "best_epoch"]
        if not all(np.isfinite(value) for value in values):
            return None
        if any(value < 0 or value > 1 for value in values[:3]) or values[3] < 0:
            return None
        return EvaluationSummary(
            test_accuracy=float(data["test_accuracy"]),
            macro_f1=float(data["macro_f1"]),
            weighted_f1=float(data["weighted_f1"]),
            best_epoch=int(data["best_epoch"]),
            val_loss=float(data["val_loss"])
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _read_dataset_summary():
    path = ARTIFACTS_DIR / "dataset_split_summary.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check the checkpoint and scaler artifacts."
        )
    architecture = type(engine.model).__name__
    return ModelInfoResponse(
        model_name=engine.model_metadata["model_name"],
        model_architecture=architecture,
        parameters_count=engine.model_metadata["parameters_count"],
        num_classes=len(engine.classes),
        classes=engine.classes,
        input_spec=InputSpec(
            num_channels=6,
            channels=CHANNELS,
            window_length=WINDOW_LENGTH,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            nominal_freq_hz=NOMINAL_FREQ_HZ,
            samples_per_cycle=SAMPLES_PER_CYCLE,
            duration_ms=round(WINDOW_LENGTH / SAMPLING_RATE_HZ * 1000.0, 2)
        ),
        evaluation_summary=_read_evaluation_summary(),
        dataset_summary=_read_dataset_summary()
    )


@app.get("/samples", response_model=SampleListResponse, tags=["Dataset"])
def get_available_samples():
    try:
        samples = [SampleIdentifier(**item) for item in engine.get_available_samples(limit=20)]
        return SampleListResponse(total_samples=len(samples), samples=samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sample list: {exc}") from exc


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    try:
        if request.sample_id is not None:
            result = engine.predict(sample_id=request.sample_id)
        else:
            waveform = np.asarray(request.waveform, dtype=np.float32)
            result = engine.predict(
                waveform=waveform,
                fault_index=request.fault_index,
                pre_fault_samples=request.pre_fault_samples
            )
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Validation error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
