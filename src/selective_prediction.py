"""
Selective Prediction (Component H).
Eq. (13): Decision = Prediction if c >= tau else 'Review Required'.
Determines threshold tau using validation data only, then evaluates on test data.
"""

import numpy as np

class SelectivePredictor:
    def __init__(self, target_risk: float = 0.05):
        self.target_risk = target_risk
        self.threshold = 0.5  # default

    def fit_threshold(self, val_probs: np.ndarray, val_labels: np.ndarray, candidate_taus=None):
        """
        Finds the lowest confidence threshold tau on validation data such that
        selective risk (error rate on accepted cases) <= target_risk.
        """
        if candidate_taus is None:
            candidate_taus = np.linspace(0.33, 0.99, 100)

        confidences = np.max(val_probs, axis=1)
        preds = np.argmax(val_probs, axis=1)

        best_tau = 0.99
        for tau in candidate_taus:
            accepted = confidences >= tau
            if accepted.sum() > 0:
                risk = np.mean(preds[accepted] != val_labels[accepted])
                if risk <= self.target_risk:
                    best_tau = tau
                    break

        self.threshold = float(best_tau)
        print(f"[SELECTIVE] Chosen tau* for target risk {self.target_risk*100:.1f}%: {self.threshold:.4f}")
        return self.threshold

    def evaluate_policy(self, test_probs: np.ndarray, test_labels: np.ndarray, tau: float = None):
        if tau is None:
            tau = self.threshold

        confidences = np.max(test_probs, axis=1)
        preds = np.argmax(test_probs, axis=1)

        accepted = confidences >= tau
        coverage = float(np.mean(accepted))
        num_accepted = int(accepted.sum())
        num_rejected = int((~accepted).sum())

        if num_accepted > 0:
            selective_accuracy = float(np.mean(preds[accepted] == test_labels[accepted]))
            selective_risk = 1.0 - selective_accuracy
        else:
            selective_accuracy = 0.0
            selective_risk = 0.0

        return {
            "threshold_tau": tau,
            "target_risk": self.target_risk,
            "test_coverage": coverage,
            "test_selective_accuracy": selective_accuracy,
            "test_selective_risk": selective_risk,
            "cases_accepted": num_accepted,
            "cases_referred_to_review": num_rejected
        }
