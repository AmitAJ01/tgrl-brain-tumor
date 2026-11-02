

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

# Locate project root and data path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "results" / "test_predictions.npz"

def load_prediction_data(filepath: Path):
    """Dynamically loads raw prediction arrays and ground truth from external data cache."""
    if not filepath.exists():
        alt_path = PROJECT_ROOT / "results" / "test_predictions_cache.npz"
        if alt_path.exists():
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Calculation data store not found at {filepath}")

    data = np.load(filepath)
    y_true = data["y_true"] if "y_true" in data else data["test_targets"]
    
    # Extract prediction labels and probabilities
    if "tgrl_preds" in data and "base_preds" in data:
        tgrl_preds = data["tgrl_preds"]
        base_preds = data["base_preds"]
    else:
        tgrl_preds = np.argmax(data["tgrl_probs_test"], axis=1)
        base_preds = np.argmax(data["base_probs_test"], axis=1)

    tgrl_probs = data["tgrl_probs"] if "tgrl_probs" in data else data["tgrl_probs_test"]
    base_probs = data["base_probs"] if "base_probs" in data else data["base_probs_test"]
    weights = data["weights"] if "weights" in data else np.ones(len(y_true))
    class_names = data["class_names"].tolist() if "class_names" in data else ["Meningioma", "Glioma", "Pituitary"]

    return y_true, base_preds, tgrl_preds, base_probs, tgrl_probs, weights, class_names

def display_results():
    y_true, base_preds, tgrl_preds, base_probs, tgrl_probs, weights, class_names = load_prediction_data(DATA_PATH)
    n_samples = len(y_true)

    # Overall Accuracy
    base_acc = np.average(y_true == base_preds, weights=weights) * 100.0
    tgrl_acc = np.average(y_true == tgrl_preds, weights=weights) * 100.0

    # Per-Class Precision, Recall, F1, Support
    base_p, base_r, base_f1, _ = precision_recall_fscore_support(
        y_true, base_preds, sample_weight=weights, zero_division=0
    )
    tgrl_p, tgrl_r, tgrl_f1, _ = precision_recall_fscore_support(
        y_true, tgrl_preds, sample_weight=weights, zero_division=0
    )

    # Macro Averages
    base_macro_p = np.mean(base_p) * 100.0
    base_macro_r = np.mean(base_r) * 100.0
    base_macro_f1 = np.mean(base_f1) * 100.0

    tgrl_macro_p = np.mean(tgrl_p) * 100.0
    tgrl_macro_r = np.mean(tgrl_r) * 100.0
    tgrl_macro_f1 = np.mean(tgrl_f1) * 100.0

    # Multi-Class One-vs-Rest ROC-AUC
    base_auc = roc_auc_score(y_true, base_probs, multi_class="ovr")
    tgrl_auc = roc_auc_score(y_true, tgrl_probs, multi_class="ovr")

    
    supports = [int(np.sum(y_true == i)) for i in range(len(class_names))]


    #Overall Classification Comparison
    print("\n1. OVERALL CLASSIFICATION COMPARISON")
    print("-" * 88)
    print(f"{'Model':<28} | {'Accuracy':<10} | {'Macro-Prec':<11} | {'Macro-Rec':<10} | {'Macro-F1':<10} | {'ROC-AUC':<8}")
    print("-" * 88)
    print(f"{'Baseline ResNet18':<28} | {base_acc:6.2f}%    | {base_macro_p:6.2f}%     | {base_macro_r:6.2f}%   | {base_macro_f1:6.2f}%   | {base_auc:6.4f}")
    print(f"{'TGRL Framework':<28} | {tgrl_acc:6.2f}%    | {tgrl_macro_p:6.2f}%     | {tgrl_macro_r:6.2f}%   | {tgrl_macro_f1:6.2f}%   | {tgrl_auc:6.4f}")
    print("-" * 88)

    # Per-Class Classification Performance
    print("\n2. PER-CLASS CLASSIFICATION PERFORMANCE")
    print("-" * 88)
    print(f"{'Model':<24} | {'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 88)

    for i, cname in enumerate(class_names):
        print(f"{'Baseline ResNet18':<24} | {cname:<12} | {base_p[i]*100:6.2f}%    | {base_r[i]*100:6.2f}%    | {base_f1[i]*100:6.2f}%    | {supports[i]:<8}")
        print(f"{'TGRL Framework':<24} | {cname:<12} | {tgrl_p[i]*100:6.2f}%    | {tgrl_r[i]*100:6.2f}%    | {tgrl_f1[i]*100:6.2f}%    | {supports[i]:<8}")
        print("-" * 88)

    print("=" * 88 + "\n")

if __name__ == "__main__":
    display_results()
