"""
GridSentry AI — Configuration Module
Centralizes paths, hyperparameters, signal specifications, and class mappings.
"""

from pathlib import Path
from typing import Dict, List

# Workspace Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"

# Dataset Archives
DATASET_1_ZIP = BASE_DIR / "signals (dataset 1).zip"
DATASET_2_ZIP = BASE_DIR / "signals_with_compensation (dataset 2).zip"

# Output Artifact Paths
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"
CLASS_MAPPING_PATH = ARTIFACTS_DIR / "class_mapping.json"
SCALER_PATH = ARTIFACTS_DIR / "scaler_params.json"
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"

# Signal Specifications
SAMPLING_RATE_HZ = 15360
NOMINAL_FREQ_HZ = 60
SAMPLES_PER_CYCLE = SAMPLING_RATE_HZ // NOMINAL_FREQ_HZ  # 256
WINDOW_LENGTH = 512  # 2 full cycles (approx 33.33 ms)
NUM_CHANNELS = 6
CHANNELS = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]

# Fault Timing References
# In Dataset 1: fault inception is around t = 0.42493 s (sample ~6527)
# In Dataset 2: fault inception is around t = 0.05 s (sample 768)
D1_FAULT_INDEX = 6527
D2_FAULT_INDEX = 768
PRE_FAULT_SAMPLES = 64  # window starts 64 samples before fault inception

# Classes
STANDARD_CLASSES: List[str] = [
    "AG",   # Phase A to Ground (Single line-to-ground)
    "BG",   # Phase B to Ground
    "CG",   # Phase C to Ground
    "AB",   # Line-to-line: A and B
    "BC",   # Line-to-line: B and C
    "CA",   # Line-to-line: C and A
    "ABG",  # Double line-to-ground: A, B, Ground
    "BCG",  # Double line-to-ground: B, C, Ground
    "CAG",  # Double line-to-ground: C, A, Ground
    "ABC"   # Three-phase symmetrical fault
]

NUM_CLASSES = len(STANDARD_CLASSES)

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(STANDARD_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(STANDARD_CLASSES)}

RAW_TO_STANDARD_CLASS: Dict[str, str] = {
    "AT": "AG",
    "BT": "BG",
    "CT": "CG",
    "AB": "AB",
    "BC": "BC",
    "CA": "CA",
    "ABT": "ABG",
    "BCT": "BCG",
    "CAT": "CAG",
    "ABC": "ABC",
    # Pass-through for standard names
    "AG": "AG",
    "BG": "BG",
    "CG": "CG",
    "ABG": "ABG",
    "BCG": "BCG",
    "CAG": "CAG",
}

# Training Hyperparameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
