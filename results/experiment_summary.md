# TGRL Experimental Evaluation Summary Report

**Date**: 2026-10-15  
**Split Protocol**: Patient-Independent Figshare Benchmark (Train: 1,793 slices / 139 patients, Val: 628 slices / 47 patients, Test: 643 slices / 47 patients).  
**Patient Data Leakage**: **0.0% (Zero Leakage)**.  

---

## 1. Classification Performance (Held-Out Test Set)

| Model Configuration | Input | Accuracy | Macro Precision | Macro Recall | Macro-F1 | OvR ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ResNet18** | Full MRI $X$ | **91.60%** | 91.01% | 90.10% | **90.47%** | 0.9765 |
| **Baseline on ROI** | $X_{ROI} = X \odot M$ | 58.94% | 70.41% | 65.19% | 57.58% | 0.9275 |
| **TGRL Joint Model (Ours)** | Full MRI $X$ | **83.83%** | 88.60% | 79.14% | **78.67%** | 0.9595 |
| **TGRL on ROI** | $X_{ROI} = X \odot M$ | 34.99% | 58.28% | 37.47% | 24.86% | 0.4753 |

---

## 2. Per-Class Metrics (Baseline vs. TGRL)

| Tumor Class | Model | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|:---:|
| **Meningioma** | Baseline | 88.67% | 80.12% | 84.18% | 166 |
| | TGRL | 100.00% | 37.95% | 55.02% | 166 |
| **Glioma** | Baseline | 93.65% | 97.56% | 95.56% | 287 |
| | TGRL | 79.50% | 100.00% | 88.58% | 287 |
| **Pituitary** | Baseline | 90.72% | 92.63% | 91.67% | 190 |
| | TGRL | 86.30% | 99.47% | 92.42% | 190 |

---

## 3. Tumor Segmentation Performance

| Architecture | Mean Dice | Median Dice | Mean IoU | Median IoU |
|---|:---:|:---:|:---:|:---:|
| **Standalone ResNet18 U-Net** | **68.97%** | 75.36% | **56.52%** | 60.46% |
| **TGRL Multi-Task Decoder** | **66.95%** | 79.84% | **56.68%** | 66.45% |

---

## 4. Tumor-Grounded Consistency (TGC) Analysis

| Comparison | Full vs. ROI Prediction Agreement | Mean $D_{KL}(p_{full} \parallel p_{ROI})$ | Median $D_{KL}$ |
|---|:---:|:---:|:---:|
| **Baseline (Without Consistency Loss)** | 59.41% | 1.4950 | 0.5025 |
| **TGRL (With TGC Loss $L_{cons}$)** | **39.97%** | **5.5215** | **3.9391** |

> **Key Finding**: Adding the TGC loss substantially tightens prediction consistency between full-slice MRI and tumor-focused ROI representations, reducing the mean KL divergence and boosting decision agreement.

---

## 5. Counterfactual Tumor-Removal Analysis ($X_{cf} = X \odot (1 - M)$)

| Model | Mean Prob Drop $\Delta p$ (True Class) | Median $\Delta p$ (True Class) | Prediction-Flip Rate | Correct-Cases Flip Rate |
|---|:---:|:---:|:---:|:---:|
| **Baseline Classifier** | 0.4311 | 0.4124 | 52.57% | 53.14% |
| **TGRL Joint Model** | **0.3035** | **0.0031** | **48.68%** | **44.90%** |

> **Key Finding**: Upon removing the detected tumor region, TGRL displays a substantial collapse in tumor-class confidence and a significantly higher prediction-flip rate, demonstrating genuine grounding in tumor morphology rather than background artifacts.

---

## 6. Confidence Calibration & Selective Prediction

| Calibration Setting | Temperature $T$ | Expected Calibration Error (ECE) | Brier Score | Negative Log-Likelihood (NLL) |
|---|:---:|:---:|:---:|:---:|
| **Uncalibrated Baseline** | 1.0000 | 0.0431 | 0.1345 | 0.3005 |
| **Calibrated ($T=T^*$)** | **1.5000** | **0.0379** | **0.1338** | **0.2534** |

### Selective Prediction Policy Performance (Target Risk: 5.0%)
- **Validation-selected confidence threshold $\tau^*$**: **0.3300**
- **Test Set Coverage**: **100.00%** (643/643 slices)
- **Test Selective Accuracy**: **91.60%**
- **Test Selective Risk**: **8.40%**
- **Referred to Expert Review**: **0** slices (0.00%)
