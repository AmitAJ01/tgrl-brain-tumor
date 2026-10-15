"""
Training Script for Baseline Classifier (Experiment 1).
Trains ResNet18 classifier with Cross-Entropy Loss on patient-split train partition.
Evaluates on validation split each epoch; saves best checkpoint.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.classifier import TumorClassifier
from src.metrics import compute_classification_metrics

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def train_baseline(epochs: int = 10, batch_size: int = 32, lr: float = 3e-4, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print(f"[BASELINE] Starting Baseline Classifier Training (Epochs={epochs}, BatchSize={batch_size}, LR={lr})")
    train_loader, val_loader, test_loader = get_dataloaders(
        cache_dir=str(PROJECT_ROOT / "data" / "cache"),
        batch_size=batch_size,
        num_workers=0
    )

    model = TumorClassifier(backbone="resnet18", num_classes=3, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_val_f1 = 0.0
    best_epoch = 0

    start_time = time.time()
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct_train += (preds == y).sum().item()
            total_train += len(y)

        scheduler.step()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss_running = 0.0
        val_probs_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss_running += loss.item() * len(y)

                probs = torch.softmax(logits, dim=1)
                val_probs_all.append(probs.cpu().numpy())
                val_targets_all.append(y.cpu().numpy())

        val_probs = np.concatenate(val_probs_all, axis=0)
        val_targets = np.concatenate(val_targets_all, axis=0)
        val_loss = val_loss_running / len(val_targets)
        val_metrics = compute_classification_metrics(val_targets, val_probs)

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        ep_duration = time.time() - ep_start
        print(f"  Epoch {ep:2d}/{epochs:2d} [{ep_duration:4.1f}s] - Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%")

        history["epoch"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "history": history
            }, CHECKPOINTS_DIR / "best_baseline_classifier.pth")

    total_duration = time.time() - start_time
    print(f"[BASELINE] Training completed in {total_duration/60:.2f} mins. Best Val F1: {best_val_f1*100:.2f}% at epoch {best_epoch}")

    # Save final model & history
    torch.save(model.state_dict(), CHECKPOINTS_DIR / "final_baseline_classifier.pth")
    with open(RESULTS_DIR / "logs" / "baseline_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history

if __name__ == "__main__":
    train_baseline(epochs=8, batch_size=32)
