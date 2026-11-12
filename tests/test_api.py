"""
GridSentry AI — API contract tests.
"""
import json

import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

def test_health_reports_actual_model_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["status"] == "healthy"
    assert payload["device"]
    assert payload["timestamp"]

def test_model_info_does_not_invent_metrics(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["num_classes"] == 10
    assert len(payload["classes"]) == 10
    assert payload["input_spec"]["num_channels"] == 6
    assert payload["input_spec"]["window_length"] == 512
    metrics = payload["evaluation_summary"]
    if metrics is not None:
        for key in ("test_accuracy", "macro_f1", "weighted_f1"):
            assert metrics[key] is None or 0.0 <= metrics[key] <= 1.0

def test_samples_endpoint(client):
    response = client.get("/samples")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_samples"] == len(payload["samples"])

def test_predict_rejects_missing_or_conflicting_inputs(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={
        "sample_id": "sample.mat", "waveform": [[0.0] * 512 for _ in range(6)]
    }).status_code == 422

def test_predict_rejects_path_traversal(client):
    assert client.post("/predict", json={"sample_id": "../secret.mat"}).status_code == 422

def test_predict_rejects_ambiguous_long_waveform(client):
    waveform = [[0.0] * 1000 for _ in range(6)]
    response = client.post("/predict", json={"waveform": waveform})
    assert response.status_code == 422

def test_predict_rejects_non_finite_waveform(client):
    waveform = [[0.0] * 512 for _ in range(6)]
    waveform[0][0] = float("nan")

    response = client.post(
        "/predict",
        content=json.dumps(
            {"waveform": waveform},
            allow_nan=True
        ),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422



def test_legacy_perfect_metrics_are_not_exposed_as_verified(monkeypatch, tmp_path):
    import backend.main as api

    legacy = {
        "test_accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "best_epoch": 10,
        "val_loss": 0.0,
    }
    path = tmp_path / "legacy_metrics.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(api, "EVALUATION_METRICS_PATH", path)

    assert api._read_evaluation_summary() is None


def test_metrics_with_failed_integrity_audit_are_not_exposed(monkeypatch, tmp_path):
    import backend.main as api

    audited = {
        "test_accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "best_epoch": 10,
        "val_loss": 0.0,
        "evaluation_protocol_version": "scenario_grouped_random_v2",
        "split_integrity": {
            "scenario_disjoint": True,
            "exact_windows_disjoint": False,
        },
    }
    path = tmp_path / "failed_audit_metrics.json"
    path.write_text(json.dumps(audited), encoding="utf-8")
    monkeypatch.setattr(api, "EVALUATION_METRICS_PATH", path)

    assert api._read_evaluation_summary() is None
