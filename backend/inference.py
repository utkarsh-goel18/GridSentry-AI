"""
GridSentry AI — Inference Engine
Loads model checkpoint and scaler parameters once on startup,
processes raw waveforms or dataset sample identifiers,
and returns predictions with confidence and class probabilities.
"""

import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from ml.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    CLASS_MAPPING_PATH,
    CONFIG_PATH,
    EVALUATION_METRICS_PATH,
    DATASET_1_ZIP,
    STANDARD_CLASSES,
    WINDOW_LENGTH,
    D1_FAULT_INDEX
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
    """
    Singleton inference service for GridSentry AI fault classification.
    Reuses the exact preprocessing, windowing, and normalization logic used during training.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path or BEST_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[torch.nn.Module] = None
        self.scaler: Optional[Dict[str, Any]] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.classes: List[str] = STANDARD_CLASSES
        self.model_metadata: Dict[str, Any] = {}
        self.loader = MatlabSignalLoader()
        self._available_samples_cache: Optional[List[Dict[str, Any]]] = None

    def load(self):
        """Loads model weights, scaler, and metadata."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        print(f"Loading GridSentry AI model from: {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        model_name = checkpoint.get('model_name', '1d_cnn')
        num_classes = checkpoint.get('num_classes', len(self.classes))

        self.model = get_model(model_name, num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load scaler
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'r') as f:
                self.scaler = json.load(f)
        elif 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        else:
            raise FileNotFoundError("Normalization scaler parameters not found.")

        self.mean = np.array(self.scaler['mean'], dtype=np.float32).reshape(6, 1)
        self.std = np.array(self.scaler['std'], dtype=np.float32).reshape(6, 1)

        # Protect against zero std
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

        # Load metadata
        self.model_metadata = {
            'model_name': model_name,
            'parameters_count': self.model.count_parameters(),
            'best_epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'val_accuracy': checkpoint.get('val_accuracy', 0.0),
            'checkpoint_path': str(self.checkpoint_path)
        }
        print("Inference engine successfully loaded and ready.")

    def preprocess_waveform(self, raw_array: np.ndarray) -> np.ndarray:
        """
        Validates, orients, slices to WINDOW_LENGTH, and normalizes using training scaler.
        Input raw_array: (6, N) or (N, 6) with N >= 512
        Returns normalized array: (6, WINDOW_LENGTH)
        """
        arr = raw_array.astype(np.float32)

        # Transpose if passed as (N, 6)
        if arr.shape[0] != 6 and arr.shape[1] == 6:
            arr = arr.T

        if arr.shape[0] != 6:
            raise ValueError(f"Expected 6 channels, but got {arr.shape[0]} channels.")

        if arr.shape[1] < WINDOW_LENGTH:
            raise ValueError(f"Signal has {arr.shape[1]} time steps; minimum {WINDOW_LENGTH} required.")

        # If longer than WINDOW_LENGTH, take central window or first WINDOW_LENGTH
        if arr.shape[1] > WINDOW_LENGTH:
            # If recording is a full ~7681 dataset recording, center around fault inception
            if arr.shape[1] >= 6000:
                start = D1_FAULT_INDEX - 64
                end = start + WINDOW_LENGTH
                win = arr[:, start:end]
            elif arr.shape[1] >= 1500:
                start = 768 - 64
                end = start + WINDOW_LENGTH
                win = arr[:, start:end]
            else:
                # Take first WINDOW_LENGTH samples
                win = arr[:, :WINDOW_LENGTH]
        else:
            win = arr[:, :WINDOW_LENGTH]

        # Apply exact training normalization: (win - mean) / std
        normalized = (win - self.mean) / self.std
        return normalized

    def predict(
        self,
        waveform: Optional[np.ndarray] = None,
        sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs model inference on either a direct waveform or a local dataset sample identifier.
        """
        if self.model is None:
            raise RuntimeError("InferenceEngine must be loaded prior to prediction.")

        t_start = time.perf_counter()
        meta = None
        input_type = "direct_waveform"

        if sample_id is not None:
            input_type = "sample_id"
            raw_sig, meta = self.loader.load_dataset_1_sample(sample_id)
            processed_waveform = self.preprocess_waveform(raw_sig)
        elif waveform is not None:
            processed_waveform = self.preprocess_waveform(waveform)
        else:
            raise ValueError("Must provide either waveform or sample_id.")

        # Convert to PyTorch tensor (1, 6, 512)
        tensor_in = torch.from_numpy(processed_waveform).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        pred_confidence = float(probs[pred_idx])
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        class_probabilities = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }

        result = {
            'predicted_fault': pred_class,
            'fault_description': FAULT_DESCRIPTIONS.get(pred_class, f"{pred_class} Fault"),
            'confidence': round(pred_confidence, 4),
            'class_probabilities': {k: round(v, 6) for k, v in class_probabilities.items()},
            'input_type': input_type,
            'sample_metadata': meta,
            'inference_time_ms': round(latency_ms, 2)
        }
        return result

    def get_available_samples(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Returns a curated list of representative sample identifiers from the local dataset.
        Cached in memory to prevent repeated zip index scans.
        """
        if self._available_samples_cache is not None:
            return self._available_samples_cache[:limit]

        samples = []
        if DATASET_1_ZIP.exists():
            with zipfile.ZipFile(DATASET_1_ZIP, 'r') as z:
                # Find one file from each of the 10 classes
                seen_classes = set()
                for f in z.namelist():
                    if f.endswith('.mat') and 'Classifier_LOC3' in f:
                        meta = MatlabSignalLoader.parse_filename_metadata(f)
                        flt = meta.get('standard_fault')
                        if flt and flt not in seen_classes:
                            seen_classes.add(flt)
                            samples.append({
                                'sample_id': Path(f).name,
                                'fault_class': flt,
                                'description': FAULT_DESCRIPTIONS.get(flt, flt),
                                'distance_km': meta.get('distance_km', 20),
                                'resistance_ohm': meta.get('resistance_ohm', 1.0),
                                'angle_deg': meta.get('angle_deg', 0),
                                'compensation': 0.0
                            })
                            if len(seen_classes) == 10:
                                break

        self._available_samples_cache = samples
        return samples[:limit]


# Global singleton instance
engine = InferenceEngine()
