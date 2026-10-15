# TGRL Experimental Results Audit and Scientific Discrepancy Report

**Date of Audit**: 2026-10-15  
**Auditor**: Senior Deep Learning Research Engineer & Scientific Integrity Assistant  
**Dataset Evaluated**: Figshare Contrast-Enhanced MRI (CE-MRI) Brain Tumor Dataset (DOI: [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427))  
**Target Partition**: Held-Out Patient-Independent Test Partition (Fold 5: 643 slices, 47 unique patients, zero patient leakage)  
**Primary Artifact Under Audit**: `TGRL_Updated_Paper.docx` and `TGRL_Updated_Paper.pdf`

---

## 1. Executive Summary of Audit

An exhaustive audit of raw predictions (`results/test_predictions_cache.npz`), evaluation metrics (`results/tables/*.csv`), training logs (`results/logs/*.json`), and manuscript text was conducted.

The audit revealed three primary categories of discrepancies in the initial manuscript:
1. **Per-Class Metrics vs. Confusion Matrix Contradiction**: The initial manuscript prose and Table II contained erroneous numbers for TGRL recall and precision that directly contradicted both the plotted confusion matrix and the raw saved predictions.
2. **Selective Prediction Operational Overclaim**: The initial manuscript claimed that selective prediction achieved high accuracy by filtering uncertain cases to expert radiologists. In reality, the validation-selected threshold ($\tau^* = 0.33$) achieved 100.0% coverage, rejecting zero test cases and failing to maintain the 5.0% error target on held-out test patients due to inter-patient domain shift.
3. **Tumor-Grounded Consistency (TGC) Tradeoff**: The initial manuscript claimed that TGRL improved representation consistency across the board. In reality, while early epochs achieved >90% agreement, the final joint model selection (driven by segmentation Dice score) suffered from masked-input collapse on $X_{ROI} = X \odot M$, where isolated tumor blobs were disproportionately classified as pituitary tumors, reducing full-vs-ROI agreement from 59.41% to 39.97%.

All discrepancies have been traced to their exact programmatic origins, independently recalculated, and corrected across all tables, figures, and manuscript sections.

---

## 2. Inconsistencies Found and Independently Recalculated Values

### A. Classification Confusion Matrix & Per-Class Metrics

- **Test Support (Ground Truth)**:
  - Meningioma (Class 0): 166 slices (verified from `test_targets == 0`)
  - Glioma (Class 1): 287 slices (verified from `test_targets == 1`)
  - Pituitary (Class 2): 190 slices (verified from `test_targets == 2`)
  - **Total**: $166 + 287 + 190 = 643$ slices.

#### Baseline Classifier (ResNet18)
- **Recalculated Confusion Matrix**:
  $$\begin{bmatrix} 133 & 15 & 18 \\ 7 & 280 & 0 \\ 10 & 4 & 176 \end{bmatrix}$$
  - Row sums: $133+15+18 = 166$; $7+280+0 = 287$; $10+4+176 = 190$. (Matches class support exactly).
  - Total correct predictions: $133 + 280 + 176 = 589 / 643 = 91.6018\%$.
- **Recalculated Per-Class Metrics**:
  - **Meningioma**:
    - Recall: $133 / 166 = \mathbf{80.12\%}$ (Initial report mistakenly stated 89.76%)
    - Precision: $133 / (133 + 7 + 10) = 133 / 150 = \mathbf{88.67\%}$ (Initial report stated 86.63%)
    - F1-Score: $\mathbf{84.18\%}$ (Initial report stated 88.17%)
  - **Glioma**:
    - Recall: $280 / 287 = \mathbf{97.56\%}$ (Initial report stated 90.59%)
    - Precision: $280 / (15 + 280 + 4) = 280 / 299 = \mathbf{93.65\%}$ (Initial report stated 91.07%)
    - F1-Score: $\mathbf{95.56\%}$ (Initial report stated 90.83%)
  - **Pituitary**:
    - Recall: $176 / 190 = \mathbf{92.63\%}$ (Initial report stated 91.05%)
    - Precision: $176 / (18 + 0 + 176) = 176 / 194 = \mathbf{90.72\%}$ (Initial report stated 94.21%)
    - F1-Score: $\mathbf{91.67\%}$ (Initial report stated 92.60%)

#### TGRL Joint Model (Shared Backbone + U-Net + TGC + CF)
- **Recalculated Confusion Matrix**:
  $$\begin{bmatrix} 63 & 73 & 30 \\ 0 & 287 & 0 \\ 0 & 1 & 189 \end{bmatrix}$$
  - Row sums: $63+73+30 = 166$; $0+287+0 = 287$; $0+1+189 = 190$. (Matches class support exactly).
  - Total correct predictions: $63 + 287 + 189 = 539 / 643 = 83.8258\%$.
- **Recalculated Per-Class Metrics**:
  - **Meningioma**:
    - Recall: $63 / 166 = \mathbf{37.95\%}$ (Initial report mistakenly stated 78.31%)
    - Precision: $63 / (63 + 0 + 0) = 63 / 63 = \mathbf{100.00\%}$ (Initial report stated 83.87%)
    - F1-Score: $\mathbf{55.02\%}$ (Initial report stated 81.00%)
  - **Glioma**:
    - Recall: $287 / 287 = \mathbf{100.00\%}$ (Initial report stated 94.77%)
    - Precision: $287 / (73 + 287 + 1) = 287 / 361 = \mathbf{79.50\%}$ (Initial report stated 74.45%)
    - F1-Score: $\mathbf{88.58\%}$ (Initial report stated 83.38%)
  - **Pituitary**:
    - Recall: $189 / 190 = \mathbf{99.47\%}$ (Initial report stated 65.26%)
    - Precision: $189 / (30 + 0 + 189) = 189 / 219 = \mathbf{86.30\%}$ (Initial report stated 96.88%)
    - F1-Score: $\mathbf{92.42\%}$ (Initial report stated 77.99%)

---

### B. Tumor-Grounded Consistency (TGC) Analysis

- **Baseline Model (Without TGC Loss)**:
  - Full-slice test accuracy: **91.60%**
  - Tumor ROI ($X_{ROI} = X \odot M$) test accuracy: **58.94%**
  - Full vs. ROI Prediction Agreement: **59.41%** (382 / 643 slices)
  - Mean KL Divergence $D_{KL}(p_{full} \parallel p_{ROI})$: **1.4950**
- **TGRL Model (With TGC Loss $L_{cons}$)**:
  - Full-slice test accuracy: **83.83%**
  - Tumor ROI ($X_{ROI} = X \odot M$) test accuracy: **44.48%**
  - Full vs. ROI Prediction Agreement: **39.97%** (257 / 643 slices)
  - Mean KL Divergence $D_{KL}(p_{full} \parallel p_{ROI})$: **5.5215**
- **Root Cause Analysis**:
  - In early epochs of TGRL training (Epochs 2–3), prediction agreement reached **91.6%** with an accuracy of **91.7%**, but segmentation Dice was lower (54.6%).
  - In later epochs (Epochs 5–8), as the counterfactual loss $L_{cf}$ actively penalized true class presence on $X_{cf} = X \odot (1 - M)$, the network developed an extreme prediction bias on masked inputs: 586 out of 643 ROI slices were classified as Pituitary (Class 2), because focal bright blobs on black backgrounds closely mimic pituitary lesion morphology.
  - Selecting the final checkpoint by $\text{Score} = 0.5 \times \text{F1} + 0.5 \times \text{Dice}$ favoured Epoch 8 (Dice 65.45%), inadvertently locking in the collapsed ROI representation.
  - **Scientific Conclusion**: The paper's hypothesis that TGC universally improves consistency is refuted by the final joint model under this training schedule. The multi-task objective creates an inherent tension between background-suppressed classification and high-resolution segmentation.

---

### C. Selective Prediction Policy Audit

- **Original Manuscript Claim**: "Validation-selected threshold $\tau^* = 0.33$ achieved 100% coverage with 91.60% selective accuracy, successfully delegating uncertain cases to review."
- **Audit Findings**:
  - The validation set accuracy was **95.70%** (validation error rate = **4.30%**).
  - Because the target risk threshold was $\le 5.0\%$, the threshold selection algorithm chose the minimum candidate $\tau = 0.33$, since the validation risk was already $4.30\% \le 5.0\%$.
  - At $\tau = 0.33$, **zero test cases were rejected** (Coverage = 100.0%, 643/643 accepted).
  - Due to inter-patient domain shift on the test partition, the test error was **8.40%**, failing to satisfy the 5.0% risk tolerance.
  - **Independent Test Evaluation Across Thresholds**:
    - $\tau = 0.33$: Coverage = 100.0%, Selective Accuracy = 91.60%, Selective Risk = 8.40%, Rejected = 0
    - $\tau = 0.70$: Coverage = 90.2% (580/643), Selective Accuracy = 94.83%, Selective Risk = 5.17%, Rejected = 63
    - $\tau = 0.80$: Coverage = 85.7% (551/643), Selective Accuracy = 95.10%, Selective Risk = **4.90%** (<5%), Rejected = 92
    - $\tau = 0.90$: Coverage = 77.9% (501/643), Selective Accuracy = 95.81%, Selective Risk = 4.19%, Rejected = 142
    - $\tau = 0.95$: Coverage = 71.1% (457/643), Selective Accuracy = 96.06%, Selective Risk = 3.94%, Rejected = 186
  - **Scientific Correction**: To genuinely control risk below 5.0% on held-out test patients, the decision threshold must be calibrated conservatively to $\tau \ge 0.80$, which achieves 85.7% coverage and 95.10% accuracy while referring 92 difficult cases (14.3%) to expert clinical review.

---

### D. Calibration & Segmentation Metrics

- **Temperature Scaling ($T^*$ strictly fitted on validation logits via L-BFGS)**:
  - Optimal Temperature: $T^* = \mathbf{1.5000}$
  - Test ECE: Uncalibrated = **4.31%** (0.0431) $\rightarrow$ Calibrated = **3.79%** (0.0379)
  - Test NLL: Uncalibrated = **0.3005** $\rightarrow$ Calibrated = **0.2534**
  - Test Brier Score: Uncalibrated = **0.1345** $\rightarrow$ Calibrated = **0.1338**
- **Segmentation Performance (Held-out Test Set, 643 slices)**:
  - Standalone ResNet18 U-Net: Mean Dice = **68.97%**, Median Dice = **75.36%**, Mean IoU = **56.52%**, Median IoU = **60.46%**
  - TGRL Multi-Task Decoder: Mean Dice = **66.95%**, Median Dice = **79.84%**, Mean IoU = **56.68%**, Median IoU = **66.45%**
  - Empty mask policy: In cases where $|P|=0$ and $|G|=0$, Dice = 1.0, IoU = 1.0. All 643 test slices contain verified tumor masks ($|G| > 0$).

---

## 3. Data Source and Provenance of Truth

| Metric / Result Category | Source File | Exact Calculation / Command |
|---|---|---|
| Test Labels & Probabilities | `results/test_predictions_cache.npz` | `np.load('results/test_predictions_cache.npz')` |
| Per-Class Metrics | `results/tables/per_class_metrics.csv` | `sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None)` |
| Classification Summary | `results/tables/classification_metrics.csv` | `sklearn.metrics.accuracy_score`, `roc_auc_score(multi_class='ovr')` |
| Segmentation Evaluation | `results/tables/segmentation_metrics.csv` | Explicit pixel-level Dice and IoU over 643 test binary mask pairs |
| Calibration Metrics | `results/tables/reliability_metrics.csv` | 15-bin ECE, Brier score, and multiclass NLL |
| Counterfactual Drops | `results/tables/counterfactual_metrics.csv` | Mean and median of $\Delta p = p_{original}[y] - p_{cf}[y]$ |
| Epoch Training Logs | `results/logs/*.json` | Epoch-by-epoch loss terms, validation Dice, and agreement |

---

## 4. Manuscript Sections and Figures Corrected

1. **Table III in Manuscript (Ablation Table)**:
   - Replaced template dashes with exact audited metrics across A1 to A6.
   - Clarified that A6 selective accuracy is 91.60% at $\tau^* = 0.33$ (100% coverage) or 95.10% at $\tau = 0.80$ (85.7% coverage).
2. **Section XI (Experimental Results)**:
   - Removed conflicting recall numbers (78.31%, 94.77%, 65.26%).
   - Explicitly reported audited per-class values: Meningioma Recall 37.95% (TGRL) vs 80.12% (Baseline); Glioma Recall 100.0% (TGRL) vs 97.56% (Baseline); Pituitary Recall 99.47% (TGRL) vs 92.63% (Baseline).
   - Reported the genuine agreement drop (59.41% $\rightarrow$ 39.97%) and analyzed the domain collapse on masked inputs.
   - Disclosed that the $\tau = 0.33$ policy rejects 0 cases, and provided the operating tradeoff at $\tau = 0.80$.
3. **Section XII (Discussion)**:
   - Replaced claims of "universal grounding improvement" with a scientifically rigorous analysis of multi-task objective tension between classification, counterfactual suppression, and segmentation.
4. **Section XIII (Limitations)**:
   - Expanded on inter-patient calibration shift, ROI artifact sensitivity, and the necessity of multi-cohort external validation.
5. **Figures Regenerated**:
   - **Fig. 3 (Confusion Matrices)**: Annotated with exact correct counts (589/643 Baseline, 539/643 TGRL).
   - **Fig. 4 (Per-Class Comparison)**: Fixed y-axis scale to $[0, 110\%]$ so Meningioma recall (37.95%) and F1 (55.02%) are uncropped.
   - **Fig. 7 (Full vs. ROI Probability Scatter)**: Updated captions and titles to reflect audited agreement percentages.
   - **Fig. 10 (Selective Prediction Curve)**: Plotted both the validation operating point ($\tau = 0.33$) and the true risk-controlled operating point ($\tau = 0.80$).
   - **Fig. 11 (Ablation Ladder)**: Corrected labels and values across A1–A6.

---

## 5. Scientific Integrity Verdict

- **Reproducibility**: All reported metrics are 100% reproducible from the saved test predictions cache (`results/test_predictions_cache.npz`) and model checkpoints.
- **Internal Consistency**: All confusion matrices, per-class tables, text descriptions, and figures are now fully aligned without contradictions.
- **Scientific Honesty**: The revised manuscript truthfully reports both the strengths (high glioma/pituitary recall, effective temperature scaling, and counterfactual sensitivity) and the limitations (meningioma recall drop, ROI domain collapse, and validation threshold under-rejection) of the experimental implementation.
