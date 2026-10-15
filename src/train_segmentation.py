"""
Training Script for Tumor Segmentation U-Net (Experiment 2).
Trains ResNet18 U-Net using compound Dice + BCE Loss on patient-split train partition.
Evaluates on validation split; saves best checkpoint.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.segmentation import ResNetUNet
from src.losses import SegmentationLoss
from src.metrics import compute_segmentation_metrics

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def train_segmentation(epochs: int = 8, batch_size: int = 32, lr: float = 3e-4, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print(f"[SEGMENTATION] Starting U-Net Training (Epochs={epochs}, BatchSize={batch_size}, LR={lr})")
    train_loader, val_loader, test_loader = get_dataloaders(
        cache_dir=str(PROJECT_ROOT / "data" / "cache"),
        batch_size=batch_size,
        num_workers=0
    )

    model = ResNetUNet(pretrained=True).to(device)
    criterion = SegmentationLoss(dice_weight=1.0, bce_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    best_val_dice = 0.0
    best_epoch = 0

    start_time = time.time()
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            target_mask = batch["mask"].to(device)

            optimizer.zero_grad()
            pred_mask = model(x)
            loss, _, _ = criterion(pred_mask, target_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(x)
            total_samples += len(x)

        scheduler.step()
        train_loss = running_loss / total_samples

        # Validation Phase
        model.eval()
        val_loss_running = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                target_mask = batch["mask"].to(device)

                pred_mask = model(x)
                loss, _, _ = criterion(pred_mask, target_mask)
                val_loss_running += loss.item() * len(x)

                val_preds_list.append(pred_mask.squeeze(1).cpu().numpy())
                val_targets_list.append(target_mask.squeeze(1).cpu().numpy())

        val_loss = val_loss_running / len(val_targets_list[0])
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)
        seg_metrics = compute_segmentation_metrics(val_preds, val_targets)

        val_dice = seg_metrics["mean_dice"]
        val_iou = seg_metrics["mean_iou"]

        ep_duration = time.time() - ep_start
        print(f"  Epoch {ep:2d}/{epochs:2d} [{ep_duration:4.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} Dice: {val_dice*100:.2f}% IoU: {val_iou*100:.2f}%")

        history["epoch"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_dice"].append(float(val_dice))
        history["val_iou"].append(float(val_iou))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": seg_metrics,
                "history": history
            }, CHECKPOINTS_DIR / "best_segmentation.pth")

    total_duration = time.time() - start_time
    print(f"[SEGMENTATION] Training completed in {total_duration/60:.2f} mins. Best Val Dice: {best_val_dice*100:.2f}% at epoch {best_epoch}")

    torch.save(model.state_dict(), CHECKPOINTS_DIR / "final_segmentation.pth")
    with open(RESULTS_DIR / "logs" / "segmentation_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history

if __name__ == "__main__":
    train_segmentation(epochs=8, batch_size=32)
