# Implementation Plan: Tumor-Grounded Reliability Learning (TGRL) for Robust Brain Tumor MRI Classification

## 1. Executive Summary & Objective
This project implements the research paper **"Tumor-Grounded Reliability Learning for Robust Brain Tumor MRI Classification"** (Mishra et al., NSUT New Delhi).
The core thesis is that brain tumor classification models must not be evaluated solely by raw accuracy; their predictions must be examined for:
1. Spatial tumor grounding (segmentation guidance and ROI consistency).
2. Behavioral sensitivity under counterfactual tumor removal ($X_{cf} = X \odot (1 - M)$).
3. Confidence calibration and risk-aware selective prediction.

This plan details the reproducible architecture, strict patient-level leakage prevention, training schedule, metric evaluation, figure generation, and manuscript updating.

---

## 2. Hardware Environment & Computational Budget
- **Platform**: Windows 11 AMD64, Python 3.10.11.
- **Compute**: Intel/AMD 12 logical processors (6 physical cores), 16 GB RAM.
- **Acceleration**: PyTorch 2.11.0 CPU (No CUDA GPU detected).
- **Optimization Strategy for CPU Execution**:
  - Image input resolution: $224 \times 224 \times 3$ (standard ImageNet resolution for pretrained backbones).
  - Backbones: Pretrained `ResNet18` and `EfficientNet-B0` (lightweight, rapid forward/backward passes on CPU with `torch.set_num_threads(8-12)`).
  - Training Epochs: 10–12 epochs with early stopping (patience 3) for baseline and segmentation; 8–10 epochs for TGRL joint training.
  - Preprocessed tensor caching / memory mapping to eliminate repeated JPEG/PNG decodes during training.

---

## 3. Dataset & Strict Patient-Level Separation
- **Dataset**: Figshare Contrast-Enhanced MRI (CE-MRI), 3,064 T1-weighted axial slices from 233 patients.
  - Meningioma: 708 slices, 82 patients.
  - Glioma: 1,426 slices, 91 patients.
  - Pituitary tumor: 930 slices, 60 patients.
- **Source**: Raw `.mat` files located in `C:\Users\amitr\Desktop\brain-tumor-ai-proj\data\raw`.
- **Patient Leakage Prevention**:
  - The dataset provides `PID` (Patient ID) for every slice and an official 5-fold partition in `cvind.mat`.
  - Inspection confirms that `cvind.mat` has **exactly 0 patient leakage** across folds: all slices for any given patient reside in one single fold.
  - **Data Partition**:
    - **Train (Folds 1, 2, 3)**: 1,793 slices, 139 patients (~58.5%).
    - **Validation (Fold 4)**: 628 slices, 47 patients (~20.5%).
    - **Test (Fold 5)**: 643 slices, 47 patients (~21.0%).
  - Zero patient overlap across splits.

---

## 4. Architectural Components

### Component A: Preprocessing & Data Pipeline
- Intensity normalization: Min-max scaling of raw 16-bit MRI intensities to $[0, 1]$, replicated to 3 channels for backbone compatibility.
- Consistent image & mask affine transformations (random horizontal flips, gentle rotations) applied identically to both image and mask during training only.
- Validation and test preprocessing is strictly deterministic (resize to $224 \times 224$, standard normalization).

### Component B: Baseline Classifier
- Pretrained ResNet18 / EfficientNet-B0 backbone with adaptive average pooling and a 3-class linear head.
- Loss: Cross-Entropy $L_{cls}$. Optimizer: AdamW ($lr=3\times 10^{-4}$, weight decay $1\times 10^{-4}$).

### Component C: Tumor Segmentation Network
- U-Net architecture with lightweight encoder (ResNet18 / EfficientNet-B0) and skip connections.
- Loss: Compound $L_{seg} = L_{Dice} + L_{BCE}$ with $\epsilon=1\times 10^{-6}$ numerical safeguard.
- Output: Tumor probability map $M = S(X) \in [0, 1]^{H \times W}$. Evaluated with test Dice and IoU.

### Component D: Tumor-Focused ROI Representation
- Tumor-focused input: $X_{ROI} = X \odot M$.
- Evaluate prediction consistency and agreement between $C(F(X))$ and $C(F(X_{ROI}))$.

### Component E: Tumor-Grounded Reliability Learning (TGRL)
- Shared backbone $F$ feeding segmentation decoder $S$ and classification head $C$.
- Tumor-Grounded Consistency loss: $L_{cons} = D_{KL}(p_{full} \parallel p_{ROI})$.
- Counterfactual training term $L_{cf}$:
  - Operational formulation: $L_{cf} = \max(0, p_{cf, y_{true}} - \delta)$ or cross-entropy towards uniform distribution / reduction of true class probability.
  - Full objective: $L = L_{cls} + \lambda_{seg} L_{seg} + \lambda_{cons} L_{cons} + \lambda_{cf} L_{cf}$. Default weights: $\lambda_{seg}=1.0, \lambda_{cons}=0.5, \lambda_{cf}=0.1$.

### Component F: Counterfactual Tumor-Removal Analysis
- Counterfactual input: $X_{cf} = X \odot (1 - M)$.
- Compute probability drop: $\Delta p = p_{original} - p_{cf}$ for predicted and ground-truth classes.
- Measure prediction-flip rate on the test set.

### Component G & H: Confidence Calibration & Selective Prediction
- Temperature scaling on validation logits: $T^* = \arg\min_T NLL(Z_{val} / T, y_{val})$.
- Metrics: Expected Calibration Error (ECE, 15 bins), Brier score, Negative Log-Likelihood (NLL).
- Selective prediction: Decision threshold $\tau \in [0.33, 0.99]$. Plot coverage vs. selective risk.

---

## 5. Planned Experiments & Milestones
1. **Milestone 1**: Dataset inspection, verification, and patient-split manifest export (`results/dataset_report.json`, `results/dataset_report.md`).
2. **Milestone 2**: Smoke test verifying tensor shapes, losses, forward/backward passes, and checkpoint saving.
3. **Milestone 3**: Baseline classification training & test evaluation.
4. **Milestone 4**: Segmentation model training & test evaluation (Dice, IoU).
5. **Milestone 5**: ROI-focused inference & prediction agreement.
6. **Milestone 6**: TGRL joint training with TGC loss & comparative evaluation against baseline.
7. **Milestone 7**: Counterfactual tumor removal stress test & sensitivity analysis.
8. **Milestone 8**: Temperature scaling calibration & selective prediction coverage-risk curves.
9. **Milestone 9**: Publication figures (11 figures in PNG/PDF) and machine-readable tables.
10. **Milestone 10**: Rebuilding and updating `TGRL_Updated_Paper.docx` with genuine experimental findings.
