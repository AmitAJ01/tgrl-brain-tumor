"""
Counterfactual Tumor Removal Analysis (Component F).
Eq. (8): X_cf = X ⊙ (1 - M).
Eq. (10): Delta p = p_original - p_cf.
Interventional stress test to examine behavioral sensitivity to tumor presence.
"""

import numpy as np
import torch
import cv2
from src.metrics import compute_counterfactual_metrics

def generate_counterfactual_batch(images: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    images: [B, 3, H, W]
    masks: [B, 1, H, W] probability maps
    """
    bin_masks = (masks >= threshold).float()
    x_cf = images * (1.0 - bin_masks)
    return x_cf

def run_counterfactual_analysis(classifier, test_loader, device="cpu", mask_threshold: float = 0.5, seg_model=None):
    """
    Runs counterfactual analysis across test dataset.
    If seg_model is provided, uses predicted masks M = S(X).
    Otherwise uses ground-truth masks.
    """
    classifier.eval()
    if seg_model is not None:
        seg_model.eval()

    all_p_full = []
    all_p_cf = []
    all_targets = []
    sample_records = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            target = batch["label"].to(device)
            
            if seg_model is not None:
                mask = seg_model(x)
            else:
                mask = batch["mask"].to(device)

            # Predictions on original image
            if hasattr(classifier, "classify_image"):
                logits_full = classifier.classify_image(x)
            else:
                logits_full = classifier(x)
            p_full = torch.softmax(logits_full, dim=1)

            # Counterfactual image: remove tumor
            x_cf = generate_counterfactual_batch(x, mask, threshold=mask_threshold)
            if hasattr(classifier, "classify_image"):
                logits_cf = classifier.classify_image(x_cf)
            else:
                logits_cf = classifier(x_cf)
            p_cf = torch.softmax(logits_cf, dim=1)

            all_p_full.append(p_full.cpu().numpy())
            all_p_cf.append(p_cf.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_p_full = np.concatenate(all_p_full, axis=0)
    all_p_cf = np.concatenate(all_p_cf, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    summary_metrics = compute_counterfactual_metrics(all_p_full, all_p_cf, all_targets)

    # Detailed per-sample records for plotting
    y_pred_full = np.argmax(all_p_full, axis=1)
    y_pred_cf = np.argmax(all_p_cf, axis=1)
    delta_true = all_p_full[np.arange(len(all_targets)), all_targets] - all_p_cf[np.arange(len(all_targets)), all_targets]
    delta_pred = all_p_full[np.arange(len(all_targets)), y_pred_full] - all_p_cf[np.arange(len(all_targets)), y_pred_full]

    return {
        "summary": summary_metrics,
        "p_full": all_p_full,
        "p_cf": all_p_cf,
        "y_true": all_targets,
        "y_pred_full": y_pred_full,
        "y_pred_cf": y_pred_cf,
        "delta_true": delta_true,
        "delta_pred": delta_pred
    }
