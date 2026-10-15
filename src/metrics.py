"""
Comprehensive evaluation metrics module for TGRL.
Computes:
  - Classification: Acc, Macro P/R/F1, Per-class P/R/F1, Confusion Matrix, OvR ROC-AUC
  - Segmentation: Mean Dice, Mean IoU, empty mask policy
  - Consistency: Full vs ROI agreement, Mean KL divergence
  - Counterfactual: Mean/Median Delta p, Prediction-Flip Rate
  - Reliability: ECE, Brier Score, NLL, Selective Risk vs Coverage
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss
)

def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names=None):
    """
    y_true: [N] true integer labels
    y_prob: [N, K] predicted class probabilities
    """
    if class_names is None:
        class_names = ["Meningioma", "Glioma", "Pituitary"]
    
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = float("nan")

    per_class_dict = {}
    for i, cname in enumerate(class_names):
        per_class_dict[cname] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(np.sum(y_true == i))
        }

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "roc_auc_ovr": float(auc),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_dict
    }

def compute_segmentation_metrics(pred_masks: np.ndarray, target_masks: np.ndarray, threshold: float = 0.5):
    """
    pred_masks: [N, H, W] in [0, 1]
    target_masks: [N, H, W] in {0, 1}
    Empty mask handling policy: If both pred and target are empty, Dice = 1.0, IoU = 1.0.
    If one is empty and other non-empty, Dice = 0.0, IoU = 0.0.
    """
    bin_pred = (pred_masks >= threshold).astype(bool)
    bin_target = (target_masks >= 0.5).astype(bool)

    dice_scores = []
    iou_scores = []

    for i in range(len(bin_pred)):
        p = bin_pred[i]
        g = bin_target[i]

        intersection = np.logical_and(p, g).sum()
        sum_pixels = p.sum() + g.sum()
        union = np.logical_or(p, g).sum()

        if sum_pixels == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = (2.0 * intersection) / sum_pixels
            iou = intersection / union if union > 0 else 0.0

        dice_scores.append(dice)
        iou_scores.append(iou)

    return {
        "mean_dice": float(np.mean(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "std_iou": float(np.std(iou_scores)),
        "median_iou": float(np.median(iou_scores)),
        "num_evaluated_samples": len(dice_scores)
    }

def compute_consistency_metrics(p_full: np.ndarray, p_roi: np.ndarray, y_true: np.ndarray):
    """
    p_full: [N, K] probabilities
    p_roi: [N, K] probabilities
    """
    pred_full = np.argmax(p_full, axis=1)
    pred_roi = np.argmax(p_roi, axis=1)

    agreement = float(np.mean(pred_full == pred_roi))
    acc_full = float(np.mean(pred_full == y_true))
    acc_roi = float(np.mean(pred_roi == y_true))

    # KL divergence per sample: sum p_full * (log p_full - log p_roi)
    p_full_safe = np.clip(p_full, 1e-7, 1.0)
    p_roi_safe = np.clip(p_roi, 1e-7, 1.0)
    kl_per_sample = np.sum(p_full_safe * (np.log(p_full_safe) - np.log(p_roi_safe)), axis=1)

    return {
        "prediction_agreement": agreement,
        "acc_full": acc_full,
        "acc_roi": acc_roi,
        "mean_kl_divergence": float(np.mean(kl_per_sample)),
        "median_kl_divergence": float(np.median(kl_per_sample))
    }

def compute_counterfactual_metrics(p_full: np.ndarray, p_cf: np.ndarray, y_true: np.ndarray):
    """
    Counterfactual tumor removal evaluation:
    Delta p = p_original - p_cf
    """
    pred_full = np.argmax(p_full, axis=1)
    pred_cf = np.argmax(p_cf, axis=1)

    correct_mask = (pred_full == y_true)
    flip_mask = (pred_full != pred_cf)

    # Prob changes for true class
    true_prob_full = p_full[np.arange(len(y_true)), y_true]
    true_prob_cf = p_cf[np.arange(len(y_true)), y_true]
    delta_true = true_prob_full - true_prob_cf

    # Prob changes for predicted class
    pred_prob_full = p_full[np.arange(len(y_true)), pred_full]
    pred_prob_cf = p_cf[np.arange(len(y_true)), pred_full]
    delta_pred = pred_prob_full - pred_prob_cf

    return {
        "all_cases": {
            "mean_delta_true_class": float(np.mean(delta_true)),
            "median_delta_true_class": float(np.median(delta_true)),
            "mean_delta_pred_class": float(np.mean(delta_pred)),
            "median_delta_pred_class": float(np.median(delta_pred)),
            "prediction_flip_rate": float(np.mean(flip_mask)),
            "total_count": len(y_true)
        },
        "correctly_classified_cases": {
            "mean_delta_true_class": float(np.mean(delta_true[correct_mask])) if correct_mask.sum() > 0 else 0.0,
            "median_delta_true_class": float(np.median(delta_true[correct_mask])) if correct_mask.sum() > 0 else 0.0,
            "prediction_flip_rate": float(np.mean(flip_mask[correct_mask])) if correct_mask.sum() > 0 else 0.0,
            "count": int(correct_mask.sum())
        }
    }

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 15):
    """
    Expected Calibration Error (ECE), Brier Score, and NLL.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin
            bin_data.append({
                "bin_idx": i,
                "range": [float(bin_boundaries[i]), float(bin_boundaries[i+1])],
                "count": int(np.sum(in_bin)),
                "accuracy": bin_acc,
                "confidence": bin_conf
            })

    # Multi-class Brier score: 1/N * sum_i sum_k (p_ik - y_ik)^2
    N, K = y_prob.shape
    y_one_hot = np.zeros((N, K))
    y_one_hot[np.arange(N), y_true] = 1.0
    brier = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))

    # Negative Log-Likelihood (NLL)
    prob_true = y_prob[np.arange(N), y_true]
    nll = float(-np.mean(np.log(np.clip(prob_true, 1e-7, 1.0))))

    return {
        "ece": float(ece),
        "brier_score": brier,
        "nll": nll,
        "bins": bin_data
    }

def compute_selective_prediction_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None):
    """
    Coverage vs Selective Risk over confidence threshold tau.
    """
    if thresholds is None:
        thresholds = np.linspace(0.33, 0.99, 50)

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)

    curve = []
    for tau in thresholds:
        accepted = confidences >= tau
        coverage = float(np.mean(accepted))
        if coverage > 0:
            selective_risk = float(np.mean(predictions[accepted] != y_true[accepted]))
        else:
            selective_risk = 0.0
        
        curve.append({
            "threshold": float(tau),
            "coverage": coverage,
            "selective_risk": selective_risk
        })

    return curve
