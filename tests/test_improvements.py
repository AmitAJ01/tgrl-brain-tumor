"""
Unit and Integration Tests for TGRL Review Improvements:
  1. 3D Volumetric Segmentation Architectures (AnisotropicUNet3D & VolumetricContextUNet).
  2. Re-balanced Tumor-Grounded Consistency (TGC) Loss (Preventing Glioma Recall Degradation).
  3. Generative Inpainting and Matched Normal-Tissue Noise Counterfactual Controls.
"""

import pytest
import sys
from pathlib import Path
import torch
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation_3d import AnisotropicUNet3D, VolumetricContextUNet, get_3d_segmentor
from src.losses import RebalancedTGCLoss, TGCLoss
from src.counterfactual import (
    inpaint_slice_cv2,
    matched_noise_slice,
    generate_counterfactual_batch,
    run_counterfactual_control_study
)
from src.tgrl_model import TGRLNetwork

def test_3d_segmentation_architectures():
    """Verify 3D U-Net forward pass, tensor dimensions, and value range [0, 1]."""
    # 1. VolumetricContextUNet with 2D slice input [B, 3, H, W]
    vol_net = VolumetricContextUNet(in_channels=3, num_depth_slices=3)
    x_2d = torch.randn(2, 3, 64, 64)
    pred_2d = vol_net(x_2d)
    assert pred_2d.shape == (2, 1, 64, 64)
    assert pred_2d.min() >= 0.0 and pred_2d.max() <= 1.0

    # 2. AnisotropicUNet3D with 3D slab input [B, 1, D, H, W]
    net_3d = AnisotropicUNet3D(in_channels=1, base_channels=8)
    x_3d = torch.randn(1, 1, 4, 32, 32)
    pred_3d = net_3d(x_3d)
    assert pred_3d.shape == (1, 1, 4, 32, 32)
    assert pred_3d.min() >= 0.0 and pred_3d.max() <= 1.0

    # 3. Factory function
    model = get_3d_segmentor("volumetric_context")
    assert isinstance(model, VolumetricContextUNet)

def test_rebalanced_tgc_loss():
    """Verify RebalancedTGCLoss prevents glioma penalization through class-specific weights & margins."""
    rebal_loss = RebalancedTGCLoss(class_weights=(1.2, 0.65, 1.15), glioma_margin=0.12)
    base_loss = TGCLoss()

    # Simulate probabilities where glioma (class 1) has slight discrepancy between full and ROI
    # e.g. p_full = [0.05, 0.90, 0.05], p_roi = [0.10, 0.80, 0.10]
    p_full = torch.tensor([[0.05, 0.90, 0.05], [0.85, 0.10, 0.05]])
    p_roi = torch.tensor([[0.10, 0.80, 0.10], [0.80, 0.15, 0.05]])
    targets = torch.tensor([1, 0])  # Sample 0 is Glioma, Sample 1 is Meningioma

    loss_val = rebal_loss(p_full, p_roi, targets=targets)
    assert torch.is_tensor(loss_val)
    assert loss_val.item() >= 0.0

    # Ensure rebalanced loss for glioma is lower than uniform TGC loss
    glioma_full = p_full[0:1]
    glioma_roi = p_roi[0:1]
    glioma_target = targets[0:1]

    l_rebal = rebal_loss(glioma_full, glioma_roi, targets=glioma_target)
    l_uniform = base_loss(glioma_full, glioma_roi)
    assert l_rebal.item() < l_uniform.item(), "Rebalanced loss must reduce over-regularization on glioma"

def test_counterfactual_inpainting_and_noise():
    """Verify generative inpainting and matched noise controls replace black mask without synthetic edge void."""
    h, w = 64, 64
    img = np.ones((h, w, 3), dtype=np.float32) * 0.5  # Neutral parenchyma background
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:44, 20:44] = 255  # Square tumor region in center

    # 1. Generative inpainting
    inpainted = inpaint_slice_cv2(img, mask)
    assert inpainted.shape == (h, w, 3)
    # The tumor region in inpainted image should NOT be zero (unlike black mask)
    assert inpainted[25, 25, 0] > 0.35, "Inpainting must fill with realistic tissue, not zero void"

    # 2. Matched noise control
    noisy = matched_noise_slice(img, mask)
    assert noisy.shape == (h, w, 3)
    assert noisy[25, 25, 0] > 0.25, "Matched noise control must synthesize parenchyma texture"

    # 3. Batch tensor generation
    t_imgs = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).repeat(2, 1, 1, 1)  # [2, 3, 64, 64]
    t_masks = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).float() / 255.0

    cf_inpaint = generate_counterfactual_batch(t_imgs, t_masks, method="inpaint")
    cf_noise = generate_counterfactual_batch(t_imgs, t_masks, method="matched_noise")
    cf_black = generate_counterfactual_batch(t_imgs, t_masks, method="black_mask")

    assert cf_inpaint.shape == t_imgs.shape
    assert cf_noise.shape == t_imgs.shape
    assert cf_black.shape == t_imgs.shape

    # Tumor center in black mask is 0.0; in inpainting and noise it is non-zero
    assert cf_black[0, :, 25, 25].sum().item() == 0.0
    assert cf_inpaint[0, :, 25, 25].sum().item() > 0.5
    assert cf_noise[0, :, 25, 25].sum().item() > 0.5

def test_tgrl_network_with_3d_and_inpainting():
    """Verify TGRLNetwork integration with 3D segmentation and inpainting forward pass."""
    net = TGRLNetwork(num_classes=3, pretrained=False, use_3d_segmentor=True)
    x = torch.randn(2, 3, 64, 64)

    outputs = net(x, compute_roi=True, compute_cf=True, cf_method="inpaint")
    assert "mask" in outputs
    assert "prob_full" in outputs
    assert "prob_roi" in outputs
    assert "prob_cf" in outputs
    assert outputs["mask"].shape == (2, 1, 64, 64)
    assert outputs["prob_full"].shape == (2, 3)
    assert outputs["prob_cf"].shape == (2, 3)
