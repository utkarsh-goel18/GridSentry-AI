import torch
import torch.nn.functional as F

from ml.calibration import fit_temperature


def test_temperature_scaling_reduces_validation_nll_for_overconfident_logits():
    # Correct class is always the largest logit, but deliberately overconfident.
    logits = torch.tensor([[12.0, 0.0], [0.0, 10.0], [9.0, 0.0], [0.0, 11.0]])
    targets = torch.tensor([0, 1, 0, 0])

    result = fit_temperature(logits, targets)

    assert result["temperature"] > 1.0
    assert result["validation_nll_after"] < result["validation_nll_before"]
    assert result["validation_samples"] == 4
    assert result["method"] == "temperature_scaling"


def test_temperature_scaling_does_not_change_argmax():
    logits = torch.tensor([[8.0, 1.0, -2.0], [-1.0, 6.0, 0.0]])
    result = fit_temperature(logits, torch.tensor([0, 1]))
    calibrated = torch.softmax(logits / result["temperature"], dim=1)

    assert torch.equal(logits.argmax(dim=1), calibrated.argmax(dim=1))


def test_temperature_scaling_rejects_empty_validation_set():
    try:
        fit_temperature(torch.empty((0, 3)), torch.empty((0,), dtype=torch.long))
    except ValueError as exc:
        assert "validation samples" in str(exc)
    else:
        raise AssertionError("Empty validation data must not be accepted.")
