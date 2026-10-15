"""
Confidence Calibration via Temperature Scaling (Component G).
Eq. (12): p_i = softmax(z_i / T).
Optimizes T using validation logits to minimize validation NLL.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.metrics import compute_calibration_metrics

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Prevent non-positive temperature
        t = torch.clamp(self.temperature, 0.05, 10.0)
        return logits / t

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 50, lr: float = 0.01):
        """
        Fit temperature T strictly on validation set.
        """
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.long)
        
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_t)
            loss = nll_criterion(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        final_temp = float(self.temperature.detach().cpu().numpy()[0])
        print(f"[CALIBRATION] Optimized Temperature T*: {final_temp:.4f}")
        return final_temp

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_t = torch.tensor(logits, dtype=torch.float32)
            scaled = self.forward(logits_t)
            probs = torch.softmax(scaled, dim=1).cpu().numpy()
        return probs
