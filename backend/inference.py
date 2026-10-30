"""
GridSentry AI — inference engine.
"""
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from ml.config import (
    BEST_MODEL_PATH, SCALER_PATH, DATASET_1_ZIP, STANDARD_CLASSES,
    WINDOW_LENGTH, D1_FAULT_INDEX, PRE_FAULT_SAMPLES
)
from ml.model import get_model
from ml.loader import MatlabSignalLoader


FAULT_DESCRIPTIONS: Dict[str, str] = {
    "AG": "Single Line-to-Ground Fault (Phase A)",
    "BG": "Single Line-to-Ground Fault (Phase B)",
    "CG": "Single Line-to-Ground Fault (Phase C)",
    "AB": "Line-to-Line Fault (Phases A and B)",
    "BC": "Line-to-Line Fault (Phases B and C)",
    "CA": "Line-to-Line Fault (Phases C and A)",
    "ABG": "Double Line-to-Ground Fault (Phases A, B, and Ground)",
    "BCG": "Double Line-to-Ground Fault (Phases B, C, and Ground)",
    "CAG": "Double Line-to-Ground Fault (Phases C, A, and Ground)",
    "ABC": "Three-Phase Symmetrical Fault (Phases A, B, and C)"
}


class InferenceEngine:
    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path or BEST_MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.scaler: Optional[Dict[str, Any]] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.classes: List[str] = list(STANDARD_CLASSES)
        self.model_metadata: Dict[str, Any] = {}
        self.loader = MatlabSignalLoader()
        self._available_samples_cache: Optional[List[Dict[str, Any]]] = None

    def load(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_name = checkpoint.get("model_name", "1d_cnn")
        classes = checkpoint.get("classes", STANDARD_CLASSES)
        if len(classes) != int(checkpoint.get("num_classes", len(classes))):
            raise ValueError("Checkpoint class list and num_classes disagree.")
        if list(classes) != list(STANDARD_CLASSES):
            raise ValueError("Checkpoint class ordering does not match the configured class mapping.")

        model = get_model(model_name, num_classes=len(classes)).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if SCALER_PATH.exists():
            with open(SCALER_PATH, "r", encoding="utf-8") as handle:
                scaler = json.load(handle)
        elif checkpoint.get("scaler") is not None:
            scaler = checkpoint["scaler"]
        else:
            raise FileNotFoundError("Normalization scaler parameters are missing.")

        mean = np.asarray(scaler.get("mean"), dtype=np.float32)
        std = np.asarray(scaler.get("std"), dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("Scaler must contain exactly six channel means and standard deviations.")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Scaler values must be finite and standard deviations positive.")

        self.model = model
        self.scaler = scaler
        self.mean = mean.reshape(6, 1)
        self.std = std.reshape(6, 1)
        self.classes = list(classes)
        self.model_metadata = {
            "model_name": model_name,
            "parameters_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "best_epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "checkpoint_path": str(self.checkpoint_path)
        }
        return self

    def preprocess_waveform(
        self,
        raw_array: np.ndarray,
        fault_index: Optional[int] = None,
        pre_fault_samples: int = PRE_FAULT_SAMPLES
    ) -> np.ndarray:
        """Orient, validate, select the requested window, then apply training scaler."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Load the inference engine before preprocessing.")
        arr = np.asarray(raw_array, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Waveform must be a two-dimensional matrix.")
        if arr.shape[0] == 6:
            pass
        elif arr.shape[1] == 6:
            arr = arr.T
        else:
            raise ValueError("Expected waveform shape (6, N) or (N, 6).")
        if arr.shape[1] < WINDOW_LENGTH:
            raise ValueError(f"Signal must contain at least {WINDOW_LENGTH} samples.")
        if not np.isfinite(arr).all():
            raise ValueError("Waveform contains NaN or infinite values.")

        if arr.shape[1] == WINDOW_LENGTH:
            window = arr
        else:
            if fault_index is None:
                raise ValueError("For waveforms longer than 512 samples, provide an explicit fault_index.")
            if not 0 <= fault_index < arr.shape[1]:
                raise ValueError("fault_index is outside the waveform.")
            start = fault_index - int(pre_fault_samples)
            end = start + WINDOW_LENGTH
            if start < 0 or end > arr.shape[1]:
                raise ValueError(
                    f"Requested window [{start}:{end}] exceeds signal bounds [0:{arr.shape[1]}]."
                )
            window = arr[:, start:end]

        normalized = (window - self.mean) / self.std
        if not np.isfinite(normalized).all():
            raise ValueError("Preprocessing produced non-finite values.")
        return normalized.astype(np.float32, copy=False)

    def predict(
        self,
        waveform: Optional[np.ndarray] = None,
        sample_id: Optional[str] = None,
        fault_index: Optional[int] = None,
        pre_fault_samples: int = PRE_FAULT_SAMPLES
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("InferenceEngine must be loaded before prediction.")
        if (sample_id is None) == (waveform is None):
            raise ValueError("Provide exactly one of sample_id or waveform.")

        started = time.perf_counter()
        metadata = None
        if sample_id is not None:
            raw_signal, metadata = self.loader.load_dataset_1_sample(sample_id)
            processed = self.preprocess_waveform(
                raw_signal, fault_index=D1_FAULT_INDEX,
                pre_fault_samples=PRE_FAULT_SAMPLES
            )
            input_type = "sample_id"
        else:
            processed = self.preprocess_waveform(
                waveform, fault_index=fault_index,
                pre_fault_samples=pre_fault_samples
            )
            input_type = "direct_waveform"

        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1).cpu().squeeze(0).numpy()

        index = int(np.argmax(probabilities))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "predicted_fault": self.classes[index],
            "fault_description": FAULT_DESCRIPTIONS.get(self.classes[index], self.classes[index]),
            "confidence": round(float(probabilities[index]), 4),
            "class_probabilities": {
                label: round(float(probabilities[i]), 6)
                for i, label in enumerate(self.classes)
            },
            "input_type": input_type,
            "sample_metadata": metadata,
            "inference_time_ms": round(elapsed_ms, 2)
        }

    def get_available_samples(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit < 1:
            return []
        if self._available_samples_cache is not None:
            return self._available_samples_cache[:limit]

        samples = []
        if DATASET_1_ZIP.exists():
            with zipfile.ZipFile(DATASET_1_ZIP, "r") as archive:
                seen = set()
                for filename in archive.namelist():
                    if not filename.endswith(".mat") or "Classifier_LOC3" not in filename:
                        continue
                    meta = MatlabSignalLoader.parse_filename_metadata(filename)
                    fault = meta.get("standard_fault")
                    if fault and fault not in seen:
                        seen.add(fault)
                        samples.append({
                            "sample_id": Path(filename).name,
                            "fault_class": fault,
                            "description": FAULT_DESCRIPTIONS.get(fault, fault),
                            "distance_km": meta.get("distance_km", 20),
                            "resistance_ohm": meta.get("resistance_ohm", 1.0),
                            "angle_deg": meta.get("angle_deg", 0),
                            "compensation": 0.0
                        })
                    if len(seen) == len(STANDARD_CLASSES):
                        break
        self._available_samples_cache = samples
        return samples[:limit]


engine = InferenceEngine()
