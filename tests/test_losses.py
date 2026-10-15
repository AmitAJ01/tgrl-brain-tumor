"""
Unit tests for TGRL Loss Functions.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import DiceLoss, SegmentationLoss, TGCLoss, OperationalCounterfactualLoss

def test_dice_loss():
    crit = DiceLoss()
    # Perfect overlap
    pred = torch.ones(2, 1, 32, 32)
    tgt = torch.ones(2, 1, 32, 32)
    loss_perf = crit(pred, tgt)
    assert abs(loss_perf.item() - 0.0) < 1e-4

    # Zero overlap
    pred_zero = torch.zeros(2, 1, 32, 32)
    tgt_one = torch.ones(2, 1, 32, 32)
    loss_zero = crit(pred_zero, tgt_one)
    assert abs(loss_zero.item() - 1.0) < 1e-2

def test_segmentation_loss():
    crit = SegmentationLoss(dice_weight=1.0, bce_weight=1.0)
    pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
    tgt = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss, l_dice, l_bce = crit(pred, tgt)
    assert loss.item() > 0.0
    assert l_dice.item() > 0.0
    assert l_bce.item() > 0.0

def test_tgc_loss():
    crit = TGCLoss()
    # Identical distributions -> KL divergence = 0
    p1 = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    loss_ident = crit(p1, p1)
    assert abs(loss_ident.item() - 0.0) < 1e-4

    # Divergent distributions -> KL divergence > 0
    p2 = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    loss_div = crit(p1, p2)
    assert loss_div.item() > 0.0

def test_counterfactual_loss():
    crit = OperationalCounterfactualLoss(delta=0.33)
    # True class probability is high after tumor removal -> penalty
    p_cf = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.85, 0.05]])
    targets = torch.tensor([0, 1])
    loss_penalized = crit(p_cf, targets)
    assert loss_penalized.item() > 0.4

    # True class probability is very low (< delta) -> zero penalty
    p_cf_low = torch.tensor([[0.1, 0.5, 0.4], [0.6, 0.2, 0.2]])
    loss_zero = crit(p_cf_low, targets)
    assert abs(loss_zero.item() - 0.0) < 1e-5

if __name__ == "__main__":
    test_dice_loss()
    test_segmentation_loss()
    test_tgc_loss()
    test_counterfactual_loss()
    print("All loss tests passed!")
