"""
GridSentry AI — preprocessing and split integrity tests.
"""
import numpy as np
from ml.loader import MatlabSignalLoader
from ml.dataset import get_scenario_splits
from ml.config import WINDOW_LENGTH
from backend.inference import InferenceEngine

def test_window_extraction_shape():
    signal = np.random.default_rng(1).normal(size=(6, 2000)).astype(np.float32)
    window = MatlabSignalLoader.extract_window(signal, 1000, 64, WINDOW_LENGTH)
    assert window.shape == (6, WINDOW_LENGTH)
    assert window.dtype == np.float32

def test_window_extraction_padding():
    signal = np.random.default_rng(2).normal(size=(6, 400)).astype(np.float32)
    window = MatlabSignalLoader.extract_window(signal, 50, 64, WINDOW_LENGTH)
    assert window.shape == (6, WINDOW_LENGTH)
    assert np.isfinite(window).all()

def test_scenario_splits_are_mutually_exclusive_and_complete():
    train, val, test = get_scenario_splits()
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 171
    assert (len(train), len(val), len(test)) == (117, 27, 27)

def _engine_with_identity_scaler():
    engine = InferenceEngine()
    engine.mean = np.zeros((6, 1), dtype=np.float32)
    engine.std = np.ones((6, 1), dtype=np.float32)
    return engine

def test_exact_window_does_not_need_fault_index():
    engine = _engine_with_identity_scaler()
    waveform = np.arange(6 * 512, dtype=np.float32).reshape(6, 512)
    np.testing.assert_array_equal(engine.preprocess_waveform(waveform), waveform)

def test_long_waveform_requires_explicit_fault_index():
    engine = _engine_with_identity_scaler()
    waveform = np.zeros((6, 1000), dtype=np.float32)
    try:
        engine.preprocess_waveform(waveform)
    except ValueError as exc:
        assert "fault_index" in str(exc)
    else:
        raise AssertionError("Long waveform without fault_index should be rejected.")

def test_explicit_fault_index_selects_expected_window():
    engine = _engine_with_identity_scaler()
    waveform = np.tile(np.arange(1000, dtype=np.float32), (6, 1))
    result = engine.preprocess_waveform(waveform, fault_index=500, pre_fault_samples=64)
    assert result.shape == (6, 512)
    np.testing.assert_array_equal(result[0], np.arange(436, 948, dtype=np.float32))

def test_inference_rejects_non_finite_values():
    engine = _engine_with_identity_scaler()
    waveform = np.zeros((6, 512), dtype=np.float32)
    waveform[0, 0] = np.nan
    try:
        engine.preprocess_waveform(waveform)
    except ValueError as exc:
        assert "NaN" in str(exc) or "infinite" in str(exc)
    else:
        raise AssertionError("Non-finite input should be rejected.")
