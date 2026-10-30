"""
GridSentry AI — Evaluation Module
Evaluates trained models on the held-out test split (unseen physical scenarios).
Computes Test Accuracy, Per-Class Precision & Recall, Macro/Weighted F1,
Confusion Matrix, and generates plots and JSON metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ml.config import (
    BEST_MODEL_PATH,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    STANDARD_CLASSES,
    NUM_CLASSES,
    BATCH_SIZE
)
from ml.dataset import get_dataloaders
from ml.model import get_model


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plots and saves normalized confusion matrix with raw counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='GridSentry AI — Test Set Confusion Matrix (Normalized)',
        ylabel='True Fault Class',
        xlabel='Predicted Fault Class'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot: {output_path}")


def evaluate(
    checkpoint_path: Path = BEST_MODEL_PATH,
    metrics_output_path: Path = EVALUATION_METRICS_PATH,
    batch_size: int = BATCH_SIZE,
    split_strategy: str = None,
    holdout_value: float = None
) -> Dict[str, Any]:
    """Runs evaluation on held-out test data and saves metrics."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {checkpoint_path} on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get('model_name', '1d_cnn')
    split_strategy = split_strategy or checkpoint.get('split_strategy', 'scenario')
    holdout_value = holdout_value if holdout_value is not None else checkpoint.get('holdout_value', 20)
    model = get_model(model_name, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, _, test_loader, scaler = get_dataloaders(
        batch_size=batch_size, split_strategy=split_strategy, holdout_value=holdout_value
    )

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"Running inference on {len(test_loader.dataset)} held-out test samples...")
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )

    per_class_metrics = {}
    for i, cls_name in enumerate(STANDARD_CLASSES):
        per_class_metrics[cls_name] = {
            'precision': float(p_class[i]),
            'recall': float(r_class[i]),
            'f1_score': float(f1_class[i]),
            'support': int(support_class[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    # Plot confusion matrix
    cm_plot_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(cm, STANDARD_CLASSES, cm_plot_path)

    metrics_result = {
        'model_name': model_name,
        'split_strategy': split_strategy,
        'holdout_value': holdout_value if split_strategy != 'scenario' else None,
        'checkpoint_path': str(checkpoint_path),
        'best_epoch': int(checkpoint.get('epoch', 0)),
        'val_loss': float(checkpoint.get('val_loss', 0.0)),
        'val_accuracy': float(checkpoint.get('val_accuracy', 0.0)),
        'test_accuracy': float(acc),
        'macro_precision': float(p_macro),
        'macro_recall': float(r_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(p_weighted),
        'weighted_recall': float(r_weighted),
        'weighted_f1': float(f1_weighted),
        'total_test_samples': int(len(y_true)),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }

    # Save metrics JSON
    metrics_output_path = Path(metrics_output_path)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_result, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_output_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"      GridSentry AI — Test Set Evaluation Summary ({model_name})")
    print("=" * 65)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro-F1 Score:    {f1_macro * 100:.2f}%")
    print(f"Weighted-F1 Score: {f1_weighted * 100:.2f}%")
    print("-" * 65)
    print(f"{'Class':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for cls_name, vals in per_class_metrics.items():
        print(f"{cls_name:<8} | {vals['precision']*100:9.2f}% | {vals['recall']*100:9.2f}% | {vals['f1_score']*100:9.2f}% | {vals['support']:<8}")
    print("=" * 65)

    return metrics_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GridSentry AI model on test set")
    parser.add_argument('--checkpoint', type=str, default=str(BEST_MODEL_PATH), help='Path to checkpoint')
    parser.add_argument('--output', type=str, default=str(EVALUATION_METRICS_PATH), help='Metrics output JSON')
    parser.add_argument('--split-strategy', choices=['scenario', 'distance', 'resistance'], default=None, help='Override checkpoint split strategy')
    parser.add_argument('--holdout-value', type=float, default=None, help='Override held-out distance or resistance')
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        metrics_output_path=args.output,
        split_strategy=args.split_strategy,
        holdout_value=args.holdout_value
    )
