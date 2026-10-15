"""
Loss functions for TGRL:
  - Segmentation: Dice Loss, Binary Cross-Entropy, Combined L_seg = L_Dice + L_BCE
  - Classification: Cross-Entropy L_cls
  - Consistency: Tumor-Grounded Consistency (TGC) L_cons = D_KL(p_full || p_ROI)
  - Counterfactual Loss: Operationalized L_cf encouraging reduction of tumor class evidence under X_cf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, 1, H, W] in [0, 1] (probabilities from sigmoid)
        target: [B, 1, H, W] in {0, 1}
        """
        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()

class SegmentationLoss(nn.Module):
    """
    Combined segmentation objective: L_seg = L_Dice + L_BCE
    as defined in Eq. (2) of the paper.
    """
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred_prob: torch.Tensor, target_mask: torch.Tensor) -> tuple:
        # Clamp pred_prob to prevent log(0) in BCE
        pred_prob_clamped = torch.clamp(pred_prob, 1e-7, 1.0 - 1e-7)
        l_dice = self.dice_loss(pred_prob, target_mask)
        l_bce = self.bce_loss(pred_prob_clamped, target_mask)
        total_seg = self.dice_weight * l_dice + self.bce_weight * l_bce
        return total_seg, l_dice, l_bce

class TGCLoss(nn.Module):
    """
    Tumor-Grounded Consistency (TGC) Loss:
    L_cons = D_KL(p_full || p_ROI) as defined in Eq. (7) of the paper.
    Encourages predicted class probabilities from full image to match predictions from ROI.
    """
    def __init__(self):
        super().__init__()

    def forward(self, p_full: torch.Tensor, p_roi: torch.Tensor) -> torch.Tensor:
        """
        p_full: [B, K] probabilities (softmax)
        p_roi: [B, K] probabilities (softmax)
        KL divergence D_KL(P || Q) = sum P * (log P - log Q)
        PyTorch kl_div(input=log Q, target=P) computes target * (log target - input).
        """
        log_p_roi = torch.log(torch.clamp(p_roi, 1e-7, 1.0))
        p_full_safe = torch.clamp(p_full, 1e-7, 1.0)
        # reduction='batchmean' mathematically aligns with D_KL
        return F.kl_div(log_p_roi, p_full_safe, reduction='batchmean')

class OperationalCounterfactualLoss(nn.Module):
    """
    Operationalized Counterfactual Loss L_cf:
    As noted in Section VI of the source paper, removing the tumor (X_cf = X ⊙ (1 - M))
    should reduce evidence for the true tumor class.
    We operationalize this by penalizing counterfactual probability for the ground truth class
    if it remains above a margin threshold delta (e.g., 1/K = 0.333 for 3 classes).
    L_cf = mean(ReLU(p_cf[y] - delta))
    """
    def __init__(self, delta: float = 0.33):
        super().__init__()
        self.delta = delta

    def forward(self, p_cf: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        p_cf: [B, K] probabilities for counterfactual images
        targets: [B] class indices
        """
        # Gather counterfactual probability for the true class
        p_cf_true = p_cf.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = F.relu(p_cf_true - self.delta).mean()
        return loss
