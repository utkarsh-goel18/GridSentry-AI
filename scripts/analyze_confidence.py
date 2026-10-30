import sys
from pathlib import Path

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from ml.config import BEST_MODEL_PATH, NUM_CLASSES
from ml.dataset import get_dataloaders
from ml.model import get_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    model = get_model(
        checkpoint.get("model_name", "1d_cnn"),
        num_classes=NUM_CLASSES,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader, _ = get_dataloaders()

    confidences = []
    margins = []
    entropies = []
    logit_ranges = []
    correct = 0
    total = 0

    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            top2 = torch.topk(probs, k=2, dim=1).values
            predictions = probs.argmax(dim=1)

            confidences.extend(top2[:, 0].cpu().numpy())
            margins.extend((top2[:, 0] - top2[:, 1]).cpu().numpy())
            entropies.extend(
                (-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1))
                .cpu()
                .numpy()
            )
            logit_ranges.extend(
                (logits.max(dim=1).values - logits.min(dim=1).values)
                .cpu()
                .numpy()
            )

            correct += (predictions == y).sum().item()
            total += y.size(0)

    confidences = np.asarray(confidences)
    margins = np.asarray(margins)
    entropies = np.asarray(entropies)
    logit_ranges = np.asarray(logit_ranges)

    print("\nCONFIDENCE ANALYSIS")
    print("=" * 45)
    print(f"Test samples: {total}")
    print(f"Accuracy: {correct / total:.4%}")
    print(f"Mean confidence: {confidences.mean():.6f}")
    print(f"Minimum confidence: {confidences.min():.6f}")
    print(f"Maximum confidence: {confidences.max():.6f}")
    print(f"Mean top-2 probability margin: {margins.mean():.6f}")
    print(f"Mean prediction entropy: {entropies.mean():.6f}")
    print(f"Mean raw logit range: {logit_ranges.mean():.6f}")
    print(
        "Samples with confidence > 0.9999: "
        f"{(confidences > 0.9999).sum()}/{total}"
    )


if __name__ == "__main__":
    main()
