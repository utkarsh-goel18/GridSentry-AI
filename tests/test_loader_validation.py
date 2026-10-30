"""Unit tests for MATLAB array orientation and signal integrity guards."""
import numpy as np
import pytest

from ml.loader import MatlabSignalLoader


def test_three_phase_time_accepts_samples_by_phase():
    raw = np.arange(30, dtype=np.float64).reshape(10, 3)
    result = MatlabSignalLoader._as_three_phase_time(raw, "V_L")
    assert result.shape == (3, 10)
    np.testing.assert_array_equal(result, raw.T)
    assert result.dtype == np.float32


def test_three_phase_time_accepts_phase_by_samples():
    raw = np.arange(30, dtype=np.float32).reshape(3, 10)
    result = MatlabSignalLoader._as_three_phase_time(raw, "V")
    assert result.shape == (3, 10)
    np.testing.assert_array_equal(result, raw)


def test_three_phase_time_rejects_unknown_axis_shape():
    with pytest.raises(ValueError, match="exactly 3"):
        MatlabSignalLoader._as_three_phase_time(np.zeros((4, 10)), "V")


def test_three_phase_time_rejects_non_finite_data():
    raw = np.zeros((10, 3), dtype=np.float32)
    raw[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        MatlabSignalLoader._as_three_phase_time(raw, "I_L")


def test_window_rejects_empty_non_overlapping_request():
    signal = np.ones((6, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="does not overlap"):
        MatlabSignalLoader.extract_window(signal, fault_index=1000, pre_fault=0, window_length=512)
