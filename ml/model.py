"""
GridSentry AI — PyTorch Model Architectures
Compact 1D Convolutional Neural Network (1D CNN) and Baseline MLP
for 10-class electrical transmission-line fault classification.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class GridSentry1DCNN(nn.Module):
    """
    Compact 1D CNN for transmission-line fault classification.
    Input: (Batch, 6, 512) -> [Va, Vb, Vc, Ia, Ib, Ic]
    Output: (Batch, num_classes) logits
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extractor: 4 Conv1D stages with BatchNorm, ReLU, and Pooling
        self.features = nn.Sequential(
            # Stage 1: Captures initial transient wavefronts across phases
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 512 -> 256

            # Stage 2: Cross-phase correlation & harmonic features
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 256 -> 128

            # Stage 3: Deeper transient pattern detection
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 -> 64

            # Stage 4: High-level fault signature representation
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # (Batch, 128, 1)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (Batch, 6, 512)
        returns: tensor of shape (Batch, num_classes)
        """
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineMLP(nn.Module):
    """
    Simple MLP baseline for comparative evaluation.
    Flattens (Batch, 6, 512) to (Batch, 3072) and passes through dense layers.
    """

    def __init__(self, in_features: int = 6 * 512, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_name: str = "1d_cnn", num_classes: int = 10) -> nn.Module:
    """Factory function for models."""
    if model_name.lower() in ("1d_cnn", "cnn", "gridsentry1dcnn"):
        return GridSentry1DCNN(in_channels=6, num_classes=num_classes)
    elif model_name.lower() in ("mlp", "baseline", "baselinemlp"):
        return BaselineMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == '__main__':
    # Test model shapes
    cnn = GridSentry1DCNN()
    mlp = BaselineMLP()
    dummy_input = torch.randn(4, 6, 512)
    cnn_out = cnn(dummy_input)
    mlp_out = mlp(dummy_input)
    print(f"GridSentry1DCNN: {cnn.count_parameters():,} parameters, output shape: {cnn_out.shape}")
    print(f"BaselineMLP:     {mlp.count_parameters():,} parameters, output shape: {mlp_out.shape}")
