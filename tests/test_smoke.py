"""
Smoke test suite for TGRL pipeline.
Verifies:
  1. Dataset loading & shapes
  2. Classifier forward, loss, backward, step
  3. Segmentation forward, loss, backward, step
  4. Joint TGRL network forward, all losses, backward, step
  5. Metrics computation (classification, segmentation, consistency, counterfactual, calibration)
  6. Checkpoint saving and loading
"""

import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.classifier import TumorClassifier
from src.segmentation import ResNetUNet
from src.tgrl_model import TGRLNetwork
from src.losses import SegmentationLoss, TGCLoss, OperationalCounterfactualLoss
from src.metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    compute_consistency_metrics,
    compute_counterfactual_metrics,
    compute_calibration_metrics
)

def run_smoke_test():
    print("=" * 60)
    print("[SMOKE TEST] Initiating TGRL Component Verification...")
    print("=" * 60)

    # 1. Dataset loading & shapes
    print("[TEST 1/6] Testing DataLoader & Batch Shapes...")
    train_loader, val_loader, test_loader = get_dataloaders(
        cache_dir=str(PROJECT_ROOT / "data" / "cache"),
        batch_size=4,
        num_workers=0
    )
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    assert images.shape == (4, 3, 224, 224), f"Expected (4, 3, 224, 224), got {images.shape}"
    assert masks.shape == (4, 1, 224, 224), f"Expected (4, 1, 224, 224), got {masks.shape}"
    assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
    print("  --> Dataset loading & tensor shapes PASSED!")

    # 2. Classifier forward, backward, step
    print("[TEST 2/6] Testing Classifier Forward, Loss & Optimization Step...")
    cls_model = TumorClassifier(backbone="resnet18", num_classes=3, pretrained=False)
    cls_opt = optim.Adam(cls_model.parameters(), lr=1e-3)
    cls_crit = nn.CrossEntropyLoss()

    cls_opt.zero_grad()
    logits = cls_model(images)
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    cls_loss = cls_crit(logits, labels)
    cls_loss.backward()
    cls_opt.step()
    print(f"  --> Classifier step PASSED! Loss = {cls_loss.item():.4f}")

    # 3. Segmentation forward, backward, step
    print("[TEST 3/6] Testing Segmentation U-Net Forward, Loss & Step...")
    seg_model = ResNetUNet(pretrained=False)
    seg_opt = optim.Adam(seg_model.parameters(), lr=1e-3)
    seg_crit = SegmentationLoss()

    seg_opt.zero_grad()
    pred_masks = seg_model(images)
    assert pred_masks.shape == (4, 1, 224, 224), f"Expected (4, 1, 224, 224), got {pred_masks.shape}"
    assert (pred_masks >= 0.0).all() and (pred_masks <= 1.0).all(), "Sigmoid outputs must be in [0, 1]"
    seg_loss, l_dice, l_bce = seg_crit(pred_masks, masks)
    seg_loss.backward()
    seg_opt.step()
    print(f"  --> Segmentation step PASSED! SegLoss = {seg_loss.item():.4f} (Dice: {l_dice.item():.4f}, BCE: {l_bce.item():.4f})")

    # 4. Joint TGRL Network forward, all losses, backward, step
    print("[TEST 4/6] Testing Joint TGRL Network Multi-Task Forward & Losses...")
    tgrl_model = TGRLNetwork(num_classes=3, pretrained=False)
    tgrl_opt = optim.Adam(tgrl_model.parameters(), lr=1e-3)
    cons_crit = TGCLoss()
    cf_crit = OperationalCounterfactualLoss(delta=0.33)

    tgrl_opt.zero_grad()
    out = tgrl_model(images, compute_roi=True, compute_cf=True)
    m = out["mask"]
    p_full = out["prob_full"]
    p_roi = out["prob_roi"]
    p_cf = out["prob_cf"]

    assert m.shape == (4, 1, 224, 224)
    assert p_full.shape == (4, 3)
    assert p_roi.shape == (4, 3)
    assert p_cf.shape == (4, 3)

    l_cls = cls_crit(out["logits_full"], labels)
    l_seg, _, _ = seg_crit(m, masks)
    l_cons = cons_crit(p_full, p_roi)
    l_cf = cf_crit(p_cf, labels)
    total_loss = l_cls + 1.0 * l_seg + 0.5 * l_cons + 0.1 * l_cf

    total_loss.backward()
    tgrl_opt.step()
    print(f"  --> TGRL multi-task step PASSED! Total: {total_loss.item():.4f} (Cls: {l_cls.item():.4f}, Seg: {l_seg.item():.4f}, Cons: {l_cons.item():.4f}, CF: {l_cf.item():.4f})")

    # 5. Metrics calculation
    print("[TEST 5/6] Testing Evaluation Metrics Modules...")
    dummy_y = np.array([0, 1, 2, 1])
    dummy_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.3, 0.6, 0.1]
    ])
    cls_m = compute_classification_metrics(dummy_y, dummy_probs)
    assert "accuracy" in cls_m and cls_m["accuracy"] == 1.0

    dummy_pred_m = np.ones((4, 224, 224))
    dummy_tgt_m = np.ones((4, 224, 224))
    seg_m = compute_segmentation_metrics(dummy_pred_m, dummy_tgt_m)
    assert abs(seg_m["mean_dice"] - 1.0) < 1e-4

    cons_m = compute_consistency_metrics(dummy_probs, dummy_probs, dummy_y)
    assert abs(cons_m["mean_kl_divergence"]) < 1e-4

    cf_m = compute_counterfactual_metrics(dummy_probs, np.roll(dummy_probs, 1, axis=1), dummy_y)
    assert "all_cases" in cf_m

    calib_m = compute_calibration_metrics(dummy_y, dummy_probs)
    assert "ece" in calib_m
    print("  --> All evaluation metrics calculation PASSED!")

    # 6. Checkpoint saving and loading
    print("[TEST 6/6] Testing Checkpoint Saving & Restoration...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_ckpt.pth"
        torch.save({"model_state_dict": tgrl_model.state_dict(), "test_metric": 0.95}, tmp_path)
        loaded = torch.load(tmp_path, map_location="cpu")
        new_model = TGRLNetwork(num_classes=3, pretrained=False)
        new_model.load_state_dict(loaded["model_state_dict"])
        assert loaded["test_metric"] == 0.95
    print("  --> Checkpoint saving and loading PASSED!")

    print("=" * 60)
    print("[SMOKE TEST] ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
