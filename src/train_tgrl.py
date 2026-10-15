"""
Training Script for Full Tumor-Grounded Reliability Learning (TGRL) Model (Experiment 4).
Jointly trains Shared Encoder F, Segmentation Decoder S, and Classification Head C
using the complete objective:
  L = L_cls + lambda_seg * L_seg + lambda_cons * L_cons + lambda_cf * L_cf
as detailed in Algorithm 1 and Eq. (11) of the paper.
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
from src.tgrl_model import TGRLNetwork
from src.losses import SegmentationLoss, TGCLoss, OperationalCounterfactualLoss
from src.metrics import compute_classification_metrics, compute_segmentation_metrics, compute_consistency_metrics

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def train_tgrl(
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    lambda_seg: float = 1.0,
    lambda_cons: float = 0.5,
    lambda_cf: float = 0.1,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print(f"[TGRL] Starting TGRL Joint Training (Epochs={epochs}, BatchSize={batch_size}, LR={lr})")
    print(f"       Loss Weights: lambda_seg={lambda_seg}, lambda_cons={lambda_cons}, lambda_cf={lambda_cf}")

    train_loader, val_loader, test_loader = get_dataloaders(
        cache_dir=str(PROJECT_ROOT / "data" / "cache"),
        batch_size=batch_size,
        num_workers=0
    )

    model = TGRLNetwork(num_classes=3, pretrained=True).to(device)

    # Optional: if baseline or segmentation checkpoints exist, we can warm-start the encoder/decoder
    seg_ckpt_path = CHECKPOINTS_DIR / "best_segmentation.pth"
    if seg_ckpt_path.exists():
        try:
            seg_data = torch.load(seg_ckpt_path, map_location="cpu")
            state = seg_data["model_state_dict"] if "model_state_dict" in seg_data else seg_data
            # Load matching decoder weights
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"       [Warm-Start] Loaded {len(matched_dict)} layers from segmentation checkpoint.")
        except Exception as e:
            print(f"       [Warm-Start Notice] {e}")

    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = SegmentationLoss(dice_weight=1.0, bce_weight=1.0)
    cons_criterion = TGCLoss()
    cf_criterion = OperationalCounterfactualLoss(delta=0.33)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "l_cls": [],
        "l_seg": [],
        "l_cons": [],
        "l_cf": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_dice": [],
        "val_agreement": []
    }

    best_combined_score = 0.0
    best_epoch = 0

    start_time = time.time()
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()

        running_loss = 0.0
        running_cls = 0.0
        running_seg = 0.0
        running_cons = 0.0
        running_cf = 0.0
        total_samples = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            target_mask = batch["mask"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward with ROI and CF representations
            outputs = model(x, compute_roi=True, compute_cf=True)
            pred_mask = outputs["mask"]
            logits_full = outputs["logits_full"]
            prob_full = outputs["prob_full"]
            prob_roi = outputs["prob_roi"]
            prob_cf = outputs["prob_cf"]

            # Compute loss components
            l_cls = cls_criterion(logits_full, y)
            l_seg, _, _ = seg_criterion(pred_mask, target_mask)
            l_cons = cons_criterion(prob_full, prob_roi)
            l_cf = cf_criterion(prob_cf, y)

            total_loss = l_cls + (lambda_seg * l_seg) + (lambda_cons * l_cons) + (lambda_cf * l_cf)

            total_loss.backward()
            optimizer.step()

            bs = len(x)
            running_loss += total_loss.item() * bs
            running_cls += l_cls.item() * bs
            running_seg += l_seg.item() * bs
            running_cons += l_cons.item() * bs
            running_cf += l_cf.item() * bs
            total_samples += bs

        scheduler.step()

        # Validation Phase
        model.eval()
        val_probs_full = []
        val_probs_roi = []
        val_masks = []
        val_targets = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                target_mask = batch["mask"].to(device)
                y = batch["label"].to(device)

                outputs = model(x, compute_roi=True, compute_cf=False)
                val_probs_full.append(outputs["prob_full"].cpu().numpy())
                val_probs_roi.append(outputs["prob_roi"].cpu().numpy())
                val_masks.append(outputs["mask"].squeeze(1).cpu().numpy())
                val_targets.append(target_mask.squeeze(1).cpu().numpy())
                val_labels.append(y.cpu().numpy())

        val_probs_full = np.concatenate(val_probs_full, axis=0)
        val_probs_roi = np.concatenate(val_probs_roi, axis=0)
        val_masks = np.concatenate(val_masks, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        cls_metrics = compute_classification_metrics(val_labels, val_probs_full)
        seg_metrics = compute_segmentation_metrics(val_masks, val_targets)
        cons_metrics = compute_consistency_metrics(val_probs_full, val_probs_roi, val_labels)

        val_acc = cls_metrics["accuracy"]
        val_f1 = cls_metrics["macro_f1"]
        val_dice = seg_metrics["mean_dice"]
        val_agree = cons_metrics["prediction_agreement"]

        # Joint model selection criterion: balanced F1 + Dice
        combined_score = 0.5 * val_f1 + 0.5 * val_dice

        ep_duration = time.time() - ep_start
        print(f"  Epoch {ep:2d}/{epochs:2d} [{ep_duration:4.1f}s] - Loss: {running_loss/total_samples:.4f} "
              f"(Cls: {running_cls/total_samples:.3f}, Seg: {running_seg/total_samples:.3f}, Cons: {running_cons/total_samples:.3f}) | "
              f"Val Acc: {val_acc*100:.1f}% F1: {val_f1*100:.1f}% Dice: {val_dice*100:.1f}% Agree: {val_agree*100:.1f}%")

        history["epoch"].append(ep)
        history["train_loss"].append(float(running_loss / total_samples))
        history["l_cls"].append(float(running_cls / total_samples))
        history["l_seg"].append(float(running_seg / total_samples))
        history["l_cons"].append(float(running_cons / total_samples))
        history["l_cf"].append(float(running_cf / total_samples))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["val_dice"].append(float(val_dice))
        history["val_agreement"].append(float(val_agree))

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cls_metrics": cls_metrics,
                "val_seg_metrics": seg_metrics,
                "val_cons_metrics": cons_metrics,
                "history": history
            }, CHECKPOINTS_DIR / "best_tgrl_model.pth")

    total_duration = time.time() - start_time
    print(f"[TGRL] Training completed in {total_duration/60:.2f} mins. Best Combined Score: {best_combined_score:.4f} at epoch {best_epoch}")

    torch.save(model.state_dict(), CHECKPOINTS_DIR / "final_tgrl_model.pth")
    with open(RESULTS_DIR / "logs" / "tgrl_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history

if __name__ == "__main__":
    train_tgrl(epochs=8, batch_size=32)
