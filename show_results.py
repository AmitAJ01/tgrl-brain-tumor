"""
TGRL Test Evaluation Results
================================================================================
Fetches raw predictions data from results/test_predictions.npz and calculates
all classification metrics (Accuracy, Macro P/R/F1, Per-Class P/R/F1, ROC-AUC)
dynamically at runtime without any hardcoded metric values.

Usage:
    python show_results.py
"""

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import show_results

if __name__ == "__main__":
    show_results.display_results()
