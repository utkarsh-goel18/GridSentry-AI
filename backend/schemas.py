"""
GridSentry AI — FastAPI Request & Response Schemas
Defines strictly validated Pydantic schemas for health, model info, and inference.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    device: str = Field(..., examples=["cpu"])
    timestamp: str = Field(..., examples=["2026-10-26T12:00:00Z"])


class InputSpec(BaseModel):
    num_channels: int = Field(6, examples=[6])
    channels: List[str] = Field(
        default=["Va", "Vb", "Vc", "Ia", "Ib", "Ic"],
        examples=[["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]]
    )
    window_length: int = Field(512, examples=[512])
    sampling_rate_hz: int = Field(15360, examples=[15360])
    nominal_freq_hz: int = Field(60, examples=[60])
    samples_per_cycle: int = Field(256, examples=[256])
    duration_ms: float = Field(33.33, examples=[33.33])


class EvaluationSummary(BaseModel):
    test_accuracy: float = Field(..., examples=[1.0])
    macro_f1: float = Field(..., examples=[1.0])
    weighted_f1: float = Field(..., examples=[1.0])
    best_epoch: int = Field(..., examples=[10])
    val_loss: float = Field(..., examples=[0.0001])


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., examples=["1d_cnn"])
    model_architecture: str = Field(..., examples=["GridSentry1DCNN (4 Conv1D blocks + BatchNorm + AdaptiveAvgPool)"])
    parameters_count: int = Field(..., examples=[111306])
    num_classes: int = Field(10, examples=[10])
    classes: List[str] = Field(..., examples=[["AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"]])
    input_spec: InputSpec
    evaluation_summary: EvaluationSummary
    dataset_summary: Dict[str, Any]


class PredictRequest(BaseModel):
    """
    Inference request accepting either:
      1. sample_id: A valid recording identifier from the local dataset.
      2. waveform: A validated 6-channel array of shape (6, 512) or (512, 6).
    """
    sample_id: Optional[str] = Field(
        None,
        description="Dataset recording filename or identifier (e.g. 'sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat')",
        examples=["sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"]
    )
    waveform: Optional[List[List[float]]] = Field(
        None,
        description="Raw 6-channel waveform. Dimensions must be either [6, N] or [N, 6] with N >= 512. "
                    "Channels order: [Va, Vb, Vc, Ia, Ib, Ic]."
    )

    @model_validator(mode='after')
    def check_inputs(self):
        sample_id = self.sample_id
        waveform = self.waveform

        if not sample_id and waveform is None:
            raise ValueError("Either 'sample_id' or 'waveform' must be provided.")
        if sample_id and waveform is not None:
            raise ValueError("Provide either 'sample_id' or 'waveform', not both.")

        if sample_id:
            # Security validation: reject directory traversal attempts
            if ".." in sample_id or "/" in sample_id or "\\" in sample_id:
                raise ValueError("sample_id must be a clean filename/identifier without path separators.")

        if waveform is not None:
            # Validate dimensions
            arr = waveform
            if len(arr) == 0:
                raise ValueError("Waveform cannot be empty.")
            # Check shape: either (6, N) with N >= 512, or (N, 6) with N >= 512
            dim1 = len(arr)
            dim2 = len(arr[0]) if dim1 > 0 else 0
            if dim1 == 6:
                if dim2 < 512:
                    raise ValueError(f"Waveform has 6 channels, but requires at least 512 time steps (got {dim2}).")
            elif dim2 == 6:
                if dim1 < 512:
                    raise ValueError(f"Waveform has 6 channels, but requires at least 512 time steps (got {dim1}).")
            else:
                raise ValueError(
                    f"Invalid waveform dimensions: [{dim1}, {dim2}]. "
                    f"Expected either [6, >=512] or [>=512, 6] corresponding to channels [Va, Vb, Vc, Ia, Ib, Ic]."
                )

        return self


class PredictResponse(BaseModel):
    predicted_fault: str = Field(..., description="Predicted fault class code", examples=["AB"])
    fault_description: str = Field(..., description="Human-readable fault name", examples=["Line-to-Line Fault (Phases A and B)"])
    confidence: float = Field(..., description="Prediction confidence (0.0 to 1.0)", examples=[0.9998])
    class_probabilities: Dict[str, float] = Field(..., description="Softmax probabilities for each of the 10 classes")
    input_type: str = Field(..., description="'sample_id' or 'direct_waveform'", examples=["sample_id"])
    sample_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata if loaded from local dataset sample")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds", examples=[1.45])


class SampleIdentifier(BaseModel):
    sample_id: str = Field(..., examples=["sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"])
    fault_class: str = Field(..., examples=["AB"])
    description: str = Field(..., examples=["Line-to-Line: A and B"])
    distance_km: int = Field(..., examples=[20])
    resistance_ohm: float = Field(..., examples=[1.0])
    angle_deg: int = Field(..., examples=[0])
    compensation: float = Field(..., examples=[0.0])


class SampleListResponse(BaseModel):
    total_samples: int = Field(..., examples=[10])
    samples: List[SampleIdentifier]
