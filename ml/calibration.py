"""Probability calibration utilities for GridSentry AI.

Temperature is fitted only on validation logits. Test data must never be used
to fit or select the temperature.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100) -> Dict[str, Any]:
    """Fit one positive scalar temperature by minimizing validation NLL."""
    logits = logits.detach().float().cpu()
    targets = targets.detach().long().cpu()
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Expected logits [N, C] and targets [N].")
    if logits.shape[0] == 0:
        raise ValueError("Cannot calibrate without validation samples.")
    if not torch.isfinite(logits).all():
        raise ValueError("Validation logits must be finite.")

    before = float(F.cross_entropy(logits, targets).item())
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0).item())
    after = float(F.cross_entropy(logits / temperature, targets).item())
    return {
        "method": "temperature_scaling",
        "temperature": temperature,
        "validation_samples": int(targets.numel()),
        "validation_nll_before": before,
        "validation_nll_after": after,
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_calibration(path: Path, checkpoint_path: Path, split_strategy: str, fitted: Dict[str, Any]) -> Dict[str, Any]:
    artifact = {
        **fitted,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "split_strategy": split_strategy,
        "fit_source": "validation_partition_only",
        "test_partition_used_for_fitting": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact
