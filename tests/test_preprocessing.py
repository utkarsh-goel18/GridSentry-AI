"""
GridSentry AI — preprocessing and split integrity tests.
"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from ml.evaluate import _split_integrity_report
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



def _tiny_partition(metadata, waveform_value):
    waveforms = torch.full((len(metadata), 6, 512), waveform_value, dtype=torch.float32)
    return SimpleNamespace(metadata=metadata, waveforms=waveforms)


def test_integrity_report_detects_scenario_overlap():
    row = {"distance_km": 20, "resistance_ohm": 1.0, "angle_deg": 0}
    train = _tiny_partition([row], 1.0)
    val = _tiny_partition([row], 2.0)
    test = _tiny_partition([], 3.0)

    report = _split_integrity_report(train, val, test)

    assert report["scenario_disjoint"] is False
    assert report["scenario_overlap_counts"]["train_validation"] == 1


def test_integrity_report_detects_exact_duplicate_windows():
    train_row = {"distance_km": 20, "resistance_ohm": 1.0, "angle_deg": 0}
    test_row = {"distance_km": 280, "resistance_ohm": 100.0, "angle_deg": 180}
    train = _tiny_partition([train_row], 1.0)
    val = _tiny_partition([], 2.0)
    test = _tiny_partition([test_row], 1.0)

    report = _split_integrity_report(train, val, test)

    assert report["scenario_disjoint"] is True
    assert report["exact_windows_disjoint"] is False
    assert report["exact_duplicate_window_counts"]["train_test"] == 1



@pytest.mark.parametrize("strategy,expected_sizes", [
    ("distance_holdout", (57, 57, 57)),
    ("resistance_holdout", (57, 57, 57)),
    ("angle_holdout", (117, 27, 27)),
])
def test_group_holdout_splits_are_disjoint_and_complete(strategy, expected_sizes):
    train, val, test = get_scenario_splits(strategy=strategy)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 171
    assert (len(train), len(val), len(test)) == expected_sizes


def test_distance_holdout_uses_unseen_distance_values():
    train, val, test = get_scenario_splits(strategy="distance_holdout")
    assert {key[0] for key in train} == {20}
    assert {key[0] for key in val} == {150}
    assert {key[0] for key in test} == {280}


def test_resistance_holdout_uses_unseen_resistance_values():
    train, val, test = get_scenario_splits(strategy="resistance_holdout")
    assert {key[1] for key in train} == {1.0}
    assert {key[1] for key in val} == {50.0}
    assert {key[1] for key in test} == {100.0}


def test_angle_holdout_uses_unseen_inception_angles():
    train, val, test = get_scenario_splits(strategy="angle_holdout")
    assert {key[2] for key in train} == set(range(0, 130, 10))
    assert {key[2] for key in val} == {130, 140, 150}
    assert {key[2] for key in test} == {160, 170, 180}
