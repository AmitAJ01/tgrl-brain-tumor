# Dataset Inspection and Patient-Level Partitioning Report

**Dataset**: Figshare CE-MRI Brain Tumor Collection (DOI: [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427))  
**Source Directory**: `C:\Users\amitr\Desktop\brain-tumor-ai-proj\data\raw`  
**Evaluation Date**: 2026-10-15  

---

## 1. Overview and Class Distribution

| Tumor Class | Class Index | Total Slices | Percentage | Total Patients | Slices / Patient (Mean) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Meningioma** | 0 | 708 | 23.11% | 82 | 8.6 |
| **Glioma** | 1 | 1426 | 46.54% | 89 | 16.0 |
| **Pituitary** | 2 | 930 | 30.35% | 62 | 15.0 |
| **Total** | - | **3064** | **100.0%** | **233** | **13.2** |

---

## 2. Image and Mask Specifications

- **Native Resolution**: (512, 512)
- **Image Data Type**: int16 (raw MRI intensities)
- **Mask Data Type**: uint8 (binary 0/1 pixel masks)
- **Unreadable Files**: 0
- **Corrupt / NaN Images**: 0
- **Invalid Mask Values**: 0
- **Empty Masks**: 0 (All slices contain confirmed tumor annotations)
- **Duplicate Image Slices**: 0 identical slice groups detected

---

## 3. Patient-Level Separation and Data Leakage Audit

> **Status: STRICT PATIENT-LEVEL SEPARATION VERIFIED (ZERO DATA LEAKAGE)**

To prevent inter-slice patient leakage, dataset partitioning follows the official 5-fold cross-validation partition (`cvind.mat`) provided with the Figshare benchmark.
- **Patients crossing fold boundaries**: **0** out of 233.
- **Train / Validation patient overlap**: **0** patients.
- **Train / Test patient overlap**: **0** patients.
- **Validation / Test patient overlap**: **0** patients.

### Partition Configuration

| Partition | Folds Included | Slices | Patients | Meningioma | Glioma | Pituitary |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | Folds 1, 2, 3 | 1793 (58.5%) | 139 (59.7%) | 418 | 814 | 561 |
| **Validation** | Fold 4 | 628 (20.5%) | 47 (20.2%) | 124 | 325 | 179 |
| **Test** | Fold 5 | 643 (21.0%) | 47 (20.2%) | 166 | 287 | 190 |
| **Total** | 5 Folds | **3,064** | **233** | **708** | **1,426** | **930** |

---

## 4. Conclusion and Compliance
The dataset satisfies all requirements specified in Section IX of the TGRL manuscript. The patient-level separation completely eliminates patient-slice leakage and establishes a rigorous benchmark for evaluating model generalization, calibration, and counterfactual sensitivity.
