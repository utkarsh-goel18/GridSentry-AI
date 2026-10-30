"""
GridSentry AI — Test Suite: Preprocessing & Data Splits
Tests window extraction, padding, scenario-level split independence,
and normalization consistency.
"""

import json
import pytest
import numpy as np

from ml.loader import MatlabSignalLoader
from ml.dataset import get_scenario_splits
from ml.config import WINDOW_LENGTH, SCALER_PATH
from backend.inference import InferenceEngine


def test_window_extraction_shape():
    """Verifies that window extraction produces exact (6, WINDOW_LENGTH) shape."""
    dummy_signal = np.random.randn(6, 2000).astype(np.float32)
    window = MatlabSignalLoader.extract_window(dummy_signal, fault_index=1000, pre_fault=64, window_length=WINDOW_LENGTH)

    assert window.shape == (6, WINDOW_LENGTH)
    assert window.dtype == np.float32


def test_window_extraction_padding():
    """Verifies that windows near edges pad appropriately without crashing."""
    dummy_signal = np.random.randn(6, 400).astype(np.float32)
    # Target window is 512, signal is only 400
    window = MatlabSignalLoader.extract_window(dummy_signal, fault_index=50, pre_fault=64, window_length=WINDOW_LENGTH)

    assert window.shape == (6, WINDOW_LENGTH)
    assert not np.isnan(window).any()


def test_scenario_splits_are_mutually_exclusive():
    """Verifies that physical simulation scenarios have ZERO leakage across Train, Val, and Test."""
    train_s, val_s, test_s = get_scenario_splits()

    assert len(train_s.intersection(val_s)) == 0, "Train and Val scenarios must not overlap!"
    assert len(train_s.intersection(test_s)) == 0, "Train and Test scenarios must not overlap!"
    assert len(val_s.intersection(test_s)) == 0, "Val and Test scenarios must not overlap!"
    assert len(train_s) + len(val_s) + len(test_s) == 171


def test_scaler_parameters_valid():
    """Verifies that scaler parameters are positive and have 6 channels."""
    assert SCALER_PATH.exists(), f"Scaler file {SCALER_PATH} must exist"
    with open(SCALER_PATH, 'r') as f:
        scaler = json.load(f)

    assert 'mean' in scaler and 'std' in scaler
    assert len(scaler['mean']) == 6
    assert len(scaler['std']) == 6
    for s in scaler['std']:
        assert s > 0, "Standard deviation must be strictly positive"


def test_inference_and_training_preprocessing_consistency():
    """Verifies that the inference engine preprocessing matches the training normalization."""
    engine = InferenceEngine()
    engine.load()

    with open(SCALER_PATH, 'r') as f:
        scaler = json.load(f)

    mean_ref = np.array(scaler['mean']).reshape(6, 1)
    std_ref = np.array(scaler['std']).reshape(6, 1)

    dummy_raw = np.ones((6, 512), dtype=np.float32) * 1000.0
    preprocessed = engine.preprocess_waveform(dummy_raw)

    expected = (dummy_raw - mean_ref) / std_ref
    np.testing.assert_allclose(preprocessed, expected, rtol=1e-5, atol=1e-5)
