"""
GridSentry AI — FastAPI Request & Response Schemas.
"""
import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

from ml.config import WINDOW_LENGTH


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class InputSpec(BaseModel):
    num_channels: int = 6
    channels: List[str] = Field(default=["Va", "Vb", "Vc", "Ia", "Ib", "Ic"])
    window_length: int = WINDOW_LENGTH
    sampling_rate_hz: int = 15360
    nominal_freq_hz: int = 60
    samples_per_cycle: int = 256
    duration_ms: float = 33.3333


class EvaluationSummary(BaseModel):
    # None means that a verified evaluation artifact is not available.
    test_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    best_epoch: Optional[int] = None
    val_loss: Optional[float] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_architecture: str
    parameters_count: int
    num_classes: int = 10
    classes: List[str]
    input_spec: InputSpec
    evaluation_summary: Optional[EvaluationSummary] = None
    dataset_summary: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    """
    Provide either a local Dataset 1 sample_id or a numeric waveform.
    Longer waveforms require an explicit zero-based fault_index.
    """
    sample_id: Optional[str] = None
    waveform: Optional[List[List[float]]] = None
    fault_index: Optional[int] = Field(
        None, ge=0,
        description="Zero-based fault inception sample in a waveform longer than 512 samples."
    )
    pre_fault_samples: int = Field(64, ge=0, le=511)

    @model_validator(mode="after")
    def check_inputs(self):
        if (self.sample_id is None) == (self.waveform is None):
            raise ValueError("Provide exactly one of 'sample_id' or 'waveform'.")

        if self.sample_id is not None:
            if not self.sample_id.strip() or ".." in self.sample_id or "/" in self.sample_id or "\\" in self.sample_id:
                raise ValueError("sample_id must be a clean filename without path separators.")
            if self.fault_index is not None:
                raise ValueError("fault_index is only valid with a direct waveform.")

        if self.waveform is not None:
            try:
                rows = len(self.waveform)
                cols = len(self.waveform[0]) if rows else 0
                if rows == 6:
                    channels, samples = rows, cols
                    if any(len(row) != samples for row in self.waveform):
                        raise ValueError("Waveform rows must have equal lengths.")
                elif cols == 6:
                    channels, samples = cols, rows
                    if any(len(row) != 6 for row in self.waveform):
                        raise ValueError("Waveform rows must have exactly 6 channels.")
                else:
                    raise ValueError("Expected waveform shape [6, N] or [N, 6].")
                if channels != 6 or samples < WINDOW_LENGTH:
                    raise ValueError(f"Waveform must contain 6 channels and at least {WINDOW_LENGTH} samples.")
                if samples > WINDOW_LENGTH and self.fault_index is None:
                    raise ValueError("Long waveforms require an explicit fault_index; automatic window guessing is disabled.")
                if self.fault_index is not None and self.fault_index >= samples:
                    raise ValueError("fault_index must be within the waveform.")
                if any(not math.isfinite(float(value)) for row in self.waveform for value in row):
                    raise ValueError("Waveform values must all be finite numbers.")
            except (TypeError, IndexError) as exc:
                raise ValueError("Waveform must be a rectangular numeric matrix.") from exc
        return self


class PredictResponse(BaseModel):
    predicted_fault: str
    fault_description: str
    confidence: float
    class_probabilities: Dict[str, float]
    input_type: str
    sample_metadata: Optional[Dict[str, Any]] = None
    inference_time_ms: float


class SampleIdentifier(BaseModel):
    sample_id: str
    fault_class: str
    description: str
    distance_km: int
    resistance_ohm: float
    angle_deg: int
    compensation: float


class SampleListResponse(BaseModel):
    total_samples: int
    samples: List[SampleIdentifier]
