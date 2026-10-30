"""
GridSentry AI — Model Training Module
Trains the 1D CNN (and optional Baseline MLP) with CrossEntropyLoss,
AdamW optimizer, ReduceLROnPlateau scheduling, early stopping,
and saves checkpoints and training history.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ml.config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    BEST_MODEL_PATH,
    CONFIG_PATH,
    RANDOM_SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_CLASSES,
    WINDOW_LENGTH,
    STANDARD_CLASSES
)
from ml.dataset import get_dataloaders
from ml.model import get_model, GridSentry1DCNN, BaselineMLP


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one training epoch, returns (epoch_loss, epoch_accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs validation, returns (val_loss, val_accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            logits = model(waveforms)
            loss = criterion(logits, labels)

            running_loss += loss.item() * waveforms.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_training_curves(history: Dict[str, List[float]], output_path: Path):
    """Plots and saves loss and accuracy history curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=1.5, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=1.5, markersize=4)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Acc', linewidth=1.5, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Val Acc', linewidth=1.5, markersize=4)
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved training history curves to: {output_path}")


def train(
    model_name: str = "1d_cnn",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_path: Path = BEST_MODEL_PATH
) -> Dict[str, Any]:
    """Complete training loop with validation, scheduling, and early stopping."""
    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on device: {device} | Model: {model_name}")

    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=batch_size)

    model = get_model(model_name, num_classes=NUM_CLASSES).to(device)
    print(f"Model architecture initialized ({model.count_parameters():,} trainable parameters)")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:6.2f}% | "
              f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc*100:6.2f}% | LR: {lr_current:.2e}")

        # Check for improvement
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_epoch = epoch
            patience_counter = 0

            # Save best model checkpoint
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': va_loss,
                'val_accuracy': va_acc,
                'model_name': model_name,
                'num_classes': NUM_CLASSES,
                'in_channels': 6,
                'window_length': WINDOW_LENGTH,
                'classes': STANDARD_CLASSES,
                'parameters_count': model.count_parameters(),
                'scaler': scaler
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Best model saved to {checkpoint_path} (Val Loss: {va_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (Best epoch: {best_epoch})")
                break

    total_time = time.time() - t0
    print(f"Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%")

    # Plot training curves
    plot_path = FIGURES_DIR / f"training_history_{model_name}.png"
    plot_training_curves(history, plot_path)

    # Save training configuration
    config_data = {
        'model_name': model_name,
        'parameters_count': model.count_parameters(),
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_accuracy': float(best_val_acc),
        'total_training_time_s': float(total_time),
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'early_stopping_patience': patience,
            'random_seed': RANDOM_SEED
        },
        'input_spec': {
            'num_channels': 6,
            'channels': ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"],
            'window_length': WINDOW_LENGTH,
            'sampling_rate_hz': 15360
        },
        'classes': STANDARD_CLASSES
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_data, f, indent=2)

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'history': history,
        'config': config_data
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GridSentry AI Model Training")
    parser.add_argument('--model', type=str, default='1d_cnn', choices=['1d_cnn', 'baseline'], help='Model type')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Custom path to save checkpoint')
    parser.add_argument('--config-out', type=str, default=None, help='Custom path to save config JSON')
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else (ARTIFACTS_DIR / f"{args.model}_model.pt" if args.model != '1d_cnn' else BEST_MODEL_PATH)
    cfg = Path(args.config_out) if args.config_out else (ARTIFACTS_DIR / f"config_{args.model}.json" if args.model != '1d_cnn' else CONFIG_PATH)

    train(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=ckpt
    )
