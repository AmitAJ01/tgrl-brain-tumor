"""
TGRL Test Evaluation Results & Accuracy Viewer
================================================================================
Fetches and formats test benchmark results from saved evaluation records.
Displays overall classification comparison and classwise precision/recall/F1.

Usage:
    python show_metrics.py
    python show_results.py
"""

import sys
from pathlib import Path

# Dispatch to scripts/show_results.py
scripts_dir = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import show_results

if __name__ == "__main__":
    show_results.display_results()
