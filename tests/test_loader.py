"""
GridSentry AI — Test Suite: MATLAB Loader and Label Extraction
Tests loading from Dataset 1 (.mat v5/v7) and Dataset 2 (.mat v7.3 HDF5),
label mapping, and metadata extraction.
"""

import pytest
import numpy as np

from ml.loader import MatlabSignalLoader
from ml.config import DATASET_1_ZIP, DATASET_2_ZIP, STANDARD_CLASSES, RAW_TO_STANDARD_CLASS


@pytest.fixture
def loader():
    return MatlabSignalLoader()


def test_loader_dataset_1_sample(loader):
    """Verifies loading a real recording from signals (dataset 1).zip."""
    sample_name = "sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"
    sig, meta = loader.load_dataset_1_sample(sample_name)

    assert isinstance(sig, np.ndarray)
    assert sig.shape == (6, 7681)
    assert sig.dtype == np.float32

    assert meta['fault_label'] == "AB"
    assert meta['distance_km'] == 20
    assert meta['resistance_ohm'] == 1.0
    assert meta['angle_deg'] == 0
    assert meta['sampling_rate'] == 15360


def test_loader_dataset_2_sample(loader):
    """Verifies loading a scenario slice from signals_with_compensation (dataset 2).zip."""
    sig, meta = loader.load_dataset_2_sample('AG', angle_idx=0, comp_idx=0, rf_idx=0, dist_idx=0)

    assert isinstance(sig, np.ndarray)
    assert sig.shape == (6, 1537)
    assert sig.dtype == np.float32

    assert meta['fault_label'] == "AG"
    assert meta['distance_km'] == 20
    assert meta['resistance_ohm'] == 1.0
    assert meta['compensation'] == 0.4
    assert meta['sampling_rate'] == 15360


def test_label_mapping():
    """Verifies all raw Portuguese fault codes correctly map to standard IEEE/IEC labels."""
    expected_mappings = {
        'AT': 'AG',
        'BT': 'BG',
        'CT': 'CG',
        'AB': 'AB',
        'BC': 'BC',
        'CA': 'CA',
        'ABT': 'ABG',
        'BCT': 'BCG',
        'CAT': 'CAG',
        'ABC': 'ABC'
    }
    for raw, expected in expected_mappings.items():
        assert RAW_TO_STANDARD_CLASS[raw] == expected
        assert RAW_TO_STANDARD_CLASS[raw] in STANDARD_CLASSES


def test_missing_file_raises_error(loader):
    """Verifies that attempting to load a non-existent recording raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load_dataset_1_sample("non_existent_fault_file_12345.mat")
