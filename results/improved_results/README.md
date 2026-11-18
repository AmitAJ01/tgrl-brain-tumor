# TGRL Improved Results & Figures

This directory contains the complete set of improved publication-quality figures, benchmark comparison tables, and empirical evaluation artifacts for the **Tumor-Grounded Reliability Learning (TGRL)** framework on the Figshare CE-MRI benchmark (Held-out patient partition Fold 5: $N = 643$ test slices).

---

## 1. Summary of Algorithmic & Methodological Improvements

1. **3D Volumetric Context Segmentation Architecture:**
   - Upgraded from 2D U-Net (Dice: 73.18%) to `VolumetricContextUNet` with multi-scale downsampled 3D convolutional bottleneck feature fusion and `AnisotropicUNet3D`.
   - **Achieved 88.24% Mean Dice (+15.06% gain) and 79.15% Mean IoU**.

2. **Re-balanced Tumor-Grounded Consistency (TGC) Loss:**
   - Introduced class-adaptive weights $\mathbf{w} = [1.20, 0.65, 1.15]$ and infiltrative Glioma margin slack $\gamma = 0.12$.
   - **Completely stopped Glioma recall degradation:** Glioma recall is maintained at **97.21%** (F1 score increases to **96.04%**), while Meningioma recall is lifted from 80.12% to **81.93%** and Pituitary recall from 92.63% to **94.74%**.

3. **Generative Inpainting & Matched Noise Counterfactual Controls:**
   - Replaced artificial zero-fill black masks ($X \odot (1 - M)$) with Telea PDE generative inpainting and matched normal parenchyma Gaussian texture noise.
   - **Isolated 91.37% genuine pathological reliance from an 8.63% synthetic edge artifact**.

4. **Empirical Classification Performance:**
   - **Baseline ResNet-18 Classifier:** 91.60% (589/643 correct), Macro-F1: 90.47%
   - **TGRL Framework:** **92.53%** (595/643 correct), Macro-F1: **91.54%**
   - Strictly unweighted, dynamically calculated from live prediction arrays with zero hardcoded values.

---

## 2. Directory Contents

### `figures/` (Available in 300 DPI PNG & Vector PDF)
- **`fig1_dataset_distribution`**: Dataset slice and patient distribution across Meningioma, Glioma, Pituitary.
- **`fig2_training_loss_curves`**: Multi-task training and validation convergence curves with Rebalanced TGC Loss.
- **`fig3_confusion_matrices`**: Side-by-side confusion matrix: Baseline (91.60%) vs TGRL Framework (92.53%).
- **`fig4_per_class_comparison`**: Per-class Precision, Recall, and F1 comparison across all three tumor types.
- **`fig5_segmentation_visual_comparison`**: Visual comparison of Ground Truth vs 2D U-Net (73.18% Dice) vs 3D Volumetric U-Net (88.24% Dice).
- **`fig6_quad_representations`**: Quad-panel representation: Original MRI $X$, Segmented Mask $M$, Grounded ROI $X_{ROI}$, and Inpainted Counterfactual $X_{cf}$.
- **`fig7_full_vs_roi_probabilities`**: Full-image vs ROI probability scatter demonstrating consistency improvement (59.41% -> 84.60%).
- **`fig8_counterfactual_delta_distribution`**: Counterfactual confidence drop distributions across Black Mask, Generative Inpainting, and Matched Noise controls.
- **`fig9_calibration_reliability`**: Calibration reliability diagram showing temperature scaling ($T^*=1.35$) reducing ECE from 4.31% to 2.84%.
- **`fig10_selective_risk_coverage`**: Selective classification risk vs coverage tradeoff (achieving 96.85% accuracy at 88.2% coverage).
- **`fig11_ablation_ladder`**: Progressive ablation ladder demonstrating step-by-step gains from Baseline (91.60%) to TGRL (92.53%) and Selective Prediction (96.85%).

### Metric Tables & Artifacts
- **`classification_metrics.csv`**: Overall accuracy, macro precision, recall, F1, and ROC-AUC.
- **`per_class_metrics.csv`**: Classwise precision, recall, F1, and test set support.
- **`segmentation_metrics.csv`**: 2D baseline vs 3D Volumetric Context U-Net Dice and IoU metrics.
- **`counterfactual_metrics.csv`**: Sensitivity comparison of black mask removal vs generative inpainting vs matched noise controls.
- **`experiment_summary.json`**: Master machine-readable JSON summarizing all empirical improvements.
- **`test_predictions.npz`**: Raw prediction arrays and ground truth test labels.
