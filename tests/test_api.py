"""
GridSentry AI — Test Suite: FastAPI Endpoints & Real Inference
Tests:
  - GET  /health
  - GET  /model-info
  - GET  /samples
  - POST /predict (real local dataset sample_id)
  - POST /predict (direct waveform)
  - POST /predict (validation & error handling)
  - End-to-end consistency
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from ml.loader import MatlabSignalLoader


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verifies that GET /health returns 200, healthy status, and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data
    assert "device" in data


def test_model_info_endpoint(client):
    """Verifies that GET /model-info returns taxonomy, parameters, input spec, and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "1d_cnn"
    assert data["num_classes"] == 10
    assert len(data["classes"]) == 10
    assert "AG" in data["classes"]
    assert "ABC" in data["classes"]
    assert data["input_spec"]["num_channels"] == 6
    assert data["input_spec"]["window_length"] == 512
    assert data["evaluation_summary"]["test_accuracy"] >= 0.95


def test_samples_endpoint(client):
    """Verifies that GET /samples returns available representative dataset samples."""
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] > 0
    sample0 = data["samples"][0]
    assert "sample_id" in sample0
    assert "fault_class" in sample0
    assert "distance_km" in sample0


def test_predict_real_sample_id(client):
    """Verifies end-to-end prediction using a real local dataset sample identifier."""
    # Test on an AB fault sample
    sample_id = "sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"
    response = client.post("/predict", json={"sample_id": sample_id})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_fault"] == "AB"
    assert data["confidence"] > 0.90
    assert "Line-to-Line" in data["fault_description"]
    assert len(data["class_probabilities"]) == 10
    assert data["input_type"] == "sample_id"
    assert data["sample_metadata"] is not None
    assert data["sample_metadata"]["fault_label"] == "AB"
    assert data["inference_time_ms"] > 0


def test_predict_direct_waveform(client):
    """Verifies prediction using a direct numerical 6-channel waveform."""
    loader = MatlabSignalLoader()
    raw_sig, _ = loader.load_dataset_1_sample("sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat")
    window_6ch = loader.extract_window(raw_sig, 6527, pre_fault=64, window_length=512)

    # Convert to nested list
    waveform_payload = window_6ch.tolist()

    response = client.post("/predict", json={"waveform": waveform_payload})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_fault"] == "AB"
    assert data["confidence"] > 0.90
    assert data["input_type"] == "direct_waveform"


def test_predict_consistency_sample_vs_waveform(client):
    """Verifies that sending a sample_id vs its extracted raw waveform yields identical prediction."""
    sample_id = "sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat"
    res_sample = client.post("/predict", json={"sample_id": sample_id}).json()

    loader = MatlabSignalLoader()
    raw_sig, _ = loader.load_dataset_1_sample(sample_id)
    # The inference engine takes start = 6527 - 64:
    win = raw_sig[:, (6527 - 64):(6527 - 64 + 512)]
    res_wave = client.post("/predict", json={"waveform": win.tolist()}).json()

    assert res_sample["predicted_fault"] == res_wave["predicted_fault"]
    # Probabilities should be virtually identical
    for cls in res_sample["class_probabilities"]:
        assert abs(res_sample["class_probabilities"][cls] - res_wave["class_probabilities"][cls]) < 1e-4


def test_predict_validation_errors(client):
    """Tests various invalid requests and ensures appropriate HTTP error codes and messages."""
    # 1. Missing both sample_id and waveform
    res = client.post("/predict", json={})
    assert res.status_code == 422 or res.status_code == 400

    # 2. Providing both sample_id and waveform
    res = client.post("/predict", json={
        "sample_id": "test.mat",
        "waveform": [[0.0]*512]*6
    })
    assert res.status_code == 422 or res.status_code == 400

    # 3. Path traversal security check
    res = client.post("/predict", json={"sample_id": "../../etc/passwd"})
    assert res.status_code == 422 or res.status_code == 400

    # 4. Wrong waveform channel count (e.g. 4 channels instead of 6)
    res = client.post("/predict", json={"waveform": [[0.0]*512]*4})
    assert res.status_code == 422 or res.status_code == 400

    # 5. Short waveform length (< 512)
    res = client.post("/predict", json={"waveform": [[0.0]*200]*6})
    assert res.status_code == 422 or res.status_code == 400

    # 6. Non-existent sample_id
    res = client.post("/predict", json={"sample_id": "non_existent_file.mat"})
    assert res.status_code == 404
