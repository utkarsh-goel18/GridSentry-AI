"""
GridSentry AI — Test Suite: Model Architecture & Checkpoint
Tests 1D CNN forward pass, checkpoint loading, parameter integrity,
and evaluation metrics validation.
"""

import json
import pytest
import torch

from ml.config import BEST_MODEL_PATH, EVALUATION_METRICS_PATH, NUM_CLASSES
from ml.model import GridSentry1DCNN, BaselineMLP, get_model


def test_1d_cnn_forward_shape():
    """Verifies that 1D CNN produces logits of shape (Batch, 10) for input (Batch, 6, 512)."""
    model = GridSentry1DCNN(in_channels=6, num_classes=NUM_CLASSES)
    model.eval()

    batch_size = 8
    x = torch.randn(batch_size, 6, 512)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_baseline_mlp_forward_shape():
    """Verifies that Baseline MLP produces logits of shape (Batch, 10)."""
    model = BaselineMLP(num_classes=NUM_CLASSES)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 6, 512)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, NUM_CLASSES)


def test_checkpoint_loads_successfully():
    """Verifies that the saved best_model.pt checkpoint loads without error."""
    assert BEST_MODEL_PATH.exists(), f"Checkpoint {BEST_MODEL_PATH} not found"
    checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')

    assert 'model_state_dict' in checkpoint
    assert 'val_accuracy' in checkpoint
    assert checkpoint['val_accuracy'] >= 0.95, "Trained validation accuracy should exceed 95%"

    model = get_model('1d_cnn', num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Forward pass with loaded weights
    dummy = torch.randn(2, 6, 512)
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=1)

    assert probs.shape == (2, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_evaluation_metrics_file():
    """Verifies that evaluation metrics are recorded and valid."""
    assert EVALUATION_METRICS_PATH.exists(), f"Evaluation metrics {EVALUATION_METRICS_PATH} not found"
    with open(EVALUATION_METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    assert 'test_accuracy' in metrics
    assert 'macro_f1' in metrics
    assert 'weighted_f1' in metrics
    assert metrics['test_accuracy'] >= 0.95
    assert len(metrics['per_class_metrics']) == 10
