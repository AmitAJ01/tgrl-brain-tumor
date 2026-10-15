"""
Unit tests for TGRL Evaluation Metrics.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    compute_consistency_metrics,
    compute_counterfactual_metrics,
    compute_calibration_metrics,
    compute_selective_prediction_curve
)

def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8]
    ])
    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert "per_class" in metrics

def test_segmentation_metrics():
    preds = np.zeros((2, 10, 10))
    targets = np.zeros((2, 10, 10))

    # Sample 0: perfect overlap
    preds[0, 2:5, 2:5] = 1.0
    targets[0, 2:5, 2:5] = 1.0

    # Sample 1: empty mask on both
    m = compute_segmentation_metrics(preds, targets)
    assert m["mean_dice"] == 1.0
    assert m["mean_iou"] == 1.0

def test_calibration_and_selective():
    y_true = np.array([0, 1, 2, 0])
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.4, 0.3, 0.3]
    ])
    calib = compute_calibration_metrics(y_true, y_prob)
    assert "ece" in calib
    assert "brier_score" in calib
    assert "nll" in calib

    curve = compute_selective_prediction_curve(y_true, y_prob)
    assert len(curve) > 0
    assert "coverage" in curve[0]

if __name__ == "__main__":
    test_classification_metrics()
    test_segmentation_metrics()
    test_calibration_and_selective()
    print("All metrics tests passed!")
