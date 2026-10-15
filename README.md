# Tumor-Grounded Reliability Learning (TGRL) for Robust Brain Tumor MRI Classification

This repository contains the complete, reproducible Python implementation of the research paper:  
**"Tumor-Grounded Reliability Learning for Robust Brain Tumor MRI Classification"**  
(Ankit Mishra, Amit Kumar, Saurabh Kumar, Naman, Mayank Garg — Netaji Subhas University of Technology, New Delhi, India).

---

## 1. Key Contributions of TGRL
Standard deep learning classifiers for brain MRI often achieve high nominal accuracy by picking up spurious background features or scanner acquisition signatures rather than true tumor pathology. TGRL addresses this by connecting:
1. **Multi-Class Tumor Classification**: Distinguishing Meningioma, Glioma, and Pituitary tumors.
2. **Tumor Segmentation**: ResNet18 U-Net predicting explicit tumor spatial probability maps $M = S(X)$.
3. **Tumor-Grounded Consistency (TGC)**: An explicit Kullback–Leibler divergence constraint $L_{cons} = D_{KL}(p_{full} \parallel p_{ROI})$ enforcing consistent predictions between the complete slice $X$ and the isolated tumor region $X_{ROI} = X \odot M$.
4. **Counterfactual Tumor-Removal Stress Test**: Quantifying behavioral sensitivity by zero-masking the detected tumor ($X_{cf} = X \odot (1 - M)$) and computing the confidence drop $\Delta p = p_{original} - p_{cf}$.
5. **Confidence Calibration & Risk-Aware Selective Prediction**: Temperature scaling ($T^*$) on validation logits and a confidence threshold ($\tau^*$) that delegates uncertain or out-of-distribution slices to expert radiologist review.

---

## 2. Dataset and Patient-Level Separation
- **Dataset**: Figshare Contrast-Enhanced MRI (CE-MRI) Brain Tumor Collection ([DOI: 10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427)).
- **Volume**: 3,064 T1-weighted axial slices across 233 patients.
  - Meningioma: 708 slices (82 patients)
  - Glioma: 1,426 slices (89 patients)
  - Pituitary: 930 slices (62 patients)
- **Data Leakage Audit**:
  - Partitioning uses the official Figshare 5-fold cross-validation partition (`cvind.mat`).
  - **Zero Patient Overlap**: Exactly 0 patients cross split boundaries.
  - **Splits**:
    - Train (Folds 1, 2, 3): 1,793 slices (139 patients)
    - Validation (Fold 4): 628 slices (47 patients)
    - Test (Fold 5): 643 slices (47 patients)

---

## 3. Project Structure
```
TGRL/
├── configs/
│   └── experiment.yaml             # Central hyperparameters and paths
├── src/
│   ├── dataset.py                  # PyTorch Dataset and DataLoader
│   ├── preprocessing.py            # Intensity normalization and joint affine augmentations
│   ├── classifier.py               # Pretrained ResNet18 baseline classifier
│   ├── segmentation.py             # ResNet18 U-Net segmentation architecture
│   ├── tgrl_model.py               # Multi-task shared encoder with TGC & CF heads
│   ├── losses.py                   # Compound Dice+BCE, TGC KL-divergence, operational L_cf
│   ├── train_classifier.py         # Baseline classifier training loop
│   ├── train_segmentation.py       # Segmentation U-Net training loop
│   ├── train_tgrl.py               # TGRL multi-task joint training loop
│   ├── counterfactual.py           # Counterfactual tumor removal analysis
│   ├── calibration.py              # Validation temperature scaling
│   ├── selective_prediction.py     # Coverage vs. selective risk policy
│   └── metrics.py                  # Classification, segmentation, calibration metrics
├── tests/
│   ├── test_smoke.py               # End-to-end smoke tests
│   ├── test_dataset.py             # Dataset and transformation unit tests
│   ├── test_losses.py              # Loss formulation unit tests
│   └── test_metrics.py             # Metric computation unit tests
├── scripts/
│   ├── inspect_dataset.py          # Dataset inspection and manifest generation
│   ├── prepare_cache.py            # Preprocessing cache generator
│   ├── run_all_experiments.py      # Master benchmark execution runner
│   ├── generate_all_figures.py     # 11 publication-quality figures
│   └── update_manuscript.py        # Populates TGRL_Updated_Paper.docx
├── results/
│   ├── figures/                    # High-res PNG (300 DPI) and PDF plots
│   ├── tables/                     # CSV result tables
│   ├── logs/                       # Training loss and metric histories
│   ├── dataset_report.json / .md   # Complete dataset audit
│   └── experiment_summary.json / .md# Measured experimental findings
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── README.md
├── TGRL_Paper.docx / .pdf          # Original submitted draft
└── TGRL_Updated_Paper.docx / .pdf  # Revised manuscript with empirical findings
```

---

## 4. Setup and Reproduction

### Environment Requirements
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)
- See `requirements.txt` for complete dependencies.

```bash
pip install -r requirements.txt
```

### Step 1: Inspect Dataset and Generate Patient-Split Manifest
```bash
python scripts/inspect_dataset.py
```

### Step 2: Build Preprocessed Data Cache
```bash
python scripts/prepare_cache.py
```

### Step 3: Run Smoke and Unit Tests
```bash
python tests/test_smoke.py
python tests/test_dataset.py
python tests/test_losses.py
python tests/test_metrics.py
```

### Step 4: Execute All Experiments
```bash
python scripts/run_all_experiments.py
```

### Step 5: Generate Publication Figures
```bash
python scripts/generate_all_figures.py
```

### Step 6: Update Manuscript
```bash
python scripts/update_manuscript.py
```

---

## 5. Measured Findings Summary
- **Baseline vs. TGRL**: Baseline achieves high accuracy on full images but drops significantly when restricted to tumor ROIs. TGRL enforces spatial grounding via TGC loss ($L_{cons}$), boosting prediction agreement between full-slice and ROI representations.
- **Counterfactual Stress Test**: Removing the detected tumor ($X_{cf} = X \odot (1 - M)$) results in a substantial collapse in tumor-class probability ($\Delta p$), proving that the TGRL model relies on authentic tumor morphology.
- **Calibration & Selective Prediction**: Temperature scaling reduces Expected Calibration Error (ECE), while selective prediction achieves high selective accuracy by referring ambiguous cases to expert clinical review.

---

## 6. Scientific Integrity & Limitations
- **No Synthetic Results**: All reported metrics and figures are derived directly from model predictions on the held-out test split.
- **2D Slice Limitations**: Slices are evaluated independently without 3D inter-slice spatial context.
- **Not a Standalone Diagnostic**: TGRL is designed as a decision-support and interpretability framework, not an autonomous medical device.
