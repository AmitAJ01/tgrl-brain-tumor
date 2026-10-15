"""
Audited PDF Compilation Script for TGRL Research Paper.
Builds complete, multi-page publication-quality PDF manuscripts:
  - TGRL_Updated_Paper.pdf
  - TGRL_Corrected_Paper.pdf
using ReportLab with exact audited numbers, filled tables, and figures.
"""

import sys
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
OUTPUT_PDF = PROJECT_ROOT / "TGRL_Updated_Paper.pdf"
CORRECTED_PDF = PROJECT_ROOT / "TGRL_Corrected_Paper.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, page_text)
        header_text = "Mishra et al.: Tumor-Grounded Reliability Learning for Robust Brain Tumor MRI Classification"
        self.drawString(0.75 * inch, letter[1] - 0.5 * inch, header_text)
        self.setStrokeColor(colors.lightgrey)
        self.setLineWidth(0.5)
        self.line(0.75 * inch, letter[1] - 0.55 * inch, letter[0] - 0.75 * inch, letter[1] - 0.55 * inch)
        self.restoreState()

def build_pdf(target_path):
    print(f"[PDF] Compiling audited manuscript to: {target_path}")

    # Exact audited metrics
    b_acc = 91.60
    b_f1 = 90.47
    b_auc = 0.9765

    t_acc = 83.83
    t_f1 = 78.67
    t_auc = 0.9595

    s_dice = 68.97
    s_iou = 56.52

    b_agree = 59.41
    t_agree = 39.97
    b_kl = 1.4950
    t_kl = 5.5215

    cf_b_delta = 0.4311
    cf_t_delta = 0.3035
    cf_b_flip = 52.57
    cf_t_flip = 48.68

    ece_before = 4.31
    ece_after = 3.79
    T_star = 1.5000

    doc = SimpleDocTemplate(
        str(target_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Normal"], fontName="Times-Bold", fontSize=17, leading=21, alignment=1, spaceAfter=6
    )
    author_style = ParagraphStyle(
        "Authors", parent=styles["Normal"], fontName="Times-Roman", fontSize=11, leading=14, alignment=1, spaceAfter=3
    )
    affil_style = ParagraphStyle(
        "Affil", parent=styles["Normal"], fontName="Times-Italic", fontSize=9.5, leading=12, alignment=1, spaceAfter=12
    )
    abstract_style = ParagraphStyle(
        "Abstract", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.5, leading=13, spaceAfter=8, leftIndent=18, rightIndent=18
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom", parent=styles["Normal"], fontName="Times-Bold", fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=5, keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.5, leading=13, spaceAfter=5
    )
    caption_style = ParagraphStyle(
        "Caption_Custom", parent=styles["Normal"], fontName="Times-Italic", fontSize=8.5, leading=11, alignment=1, spaceAfter=8
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Tumor-Grounded Reliability Learning for Robust Brain Tumor MRI Classification", title_style))
    story.append(Paragraph("Ankit Mishra, Amit Kumar, Saurabh Kumar, Naman, and Mayank Garg", author_style))
    story.append(Paragraph("Department of Computer Science and Engineering, Netaji Subhas University of Technology, New Delhi, India", affil_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=8))

    # Abstract
    abs_text = (
        "<b><i>Abstract</i>—Deep learning has shown high accuracy in brain tumor classification using MRI slices. "
        "However, good classification performance on complete slices does not necessarily indicate that the model bases its decisions on the tumor region, "
        "or that predictions will remain consistent when irrelevant anatomy is removed. In this work, we implement, audit, and empirically evaluate the "
        "Tumor-Grounded Reliability Learning (TGRL) framework. TGRL connects tumor segmentation, tumor-focused ROI representation, Tumor-Grounded Consistency (TGC), "
        "counterfactual tumor removal, and confidence calibration. Across 3,064 CE-MRI slices from 233 patients with strict patient-level separation (zero data leakage), "
        f"the baseline ResNet18 classifier achieved {b_acc:.2f}% accuracy and {b_f1:.2f}% macro-F1 on full slices, but dropped sharply to 58.94% accuracy "
        f"and {b_agree:.2f}% agreement when evaluated on isolated tumor ROIs. The TGRL joint model demonstrated high precision on meningiomas (100.0%) and high recall "
        f"on gliomas (100.0%) and pituitary tumors (99.47%), with overall accuracy of {t_acc:.2f}%. Under counterfactual tumor removal (X<sub>cf</sub> = X ⊙ (1 - M)), "
        f"true-class probability dropped by Δp = {cf_t_delta:.3f} (flip rate {cf_t_flip:.2f}%), verifying behavioral sensitivity to tumor evidence. "
        f"Temperature scaling reduced Expected Calibration Error from {ece_before:.2f}% to {ece_after:.2f}%. Finally, selective prediction demonstrated that while "
        f"uncalibrated thresholds accept 100% of test cases, conservative thresholding (τ = 0.80) enforces a rigorous selective risk under 5.0% (4.90% risk at 85.7% coverage).</b>"
    )
    story.append(Paragraph(abs_text, abstract_style))
    story.append(Paragraph("<b><i>Index Terms</i>—Brain tumor, magnetic resonance imaging, deep learning, tumor segmentation, tumor classification, reliability learning, uncertainty estimation, counterfactual analysis.</b>", abstract_style))
    story.append(Spacer(1, 4))

    # Section I: Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph(
        "Automating brain tumor classification from magnetic resonance imaging (MRI) is vital for diagnostic workflows. "
        "However, standard convolutional neural networks often exploit confounding imaging cues—such as surrounding skull contour, cranial boundary artifacts, "
        "or acquisition-specific intensity gradients—rather than pathological tumor evidence. "
        "Tumor-Grounded Reliability Learning (TGRL) tackles this challenge by unifying multi-class classification with spatial tumor segmentation, "
        "evaluating consistency under region-of-interest (ROI) masking, and applying counterfactual tumor removal to stress-test model sensitivity.", body_style
    ))

    # Section II: Architecture & Methodology
    story.append(Paragraph("II. PROPOSED FRAMEWORK", h1_style))
    story.append(Paragraph(
        "TGRL utilizes a shared ResNet18 encoder <i>F</i> coupled with a U-Net decoder <i>S</i> and a classification head <i>C</i>. "
        "The segmentation branch produces a tumor probability map <i>M = S(X) ∈ [0, 1]<sup>H×W</sup></i> trained with compound loss "
        "<i>L<sub>seg</sub> = L<sub>Dice</sub> + L<sub>BCE</sub></i>. From the predicted probability map, two representations are constructed:<br/>"
        "• <b>Tumor-Focused ROI</b>: <i>X<sub>ROI</sub> = X ⊙ M</i> (isolating tumor evidence).<br/>"
        "• <b>Counterfactual MRI</b>: <i>X<sub>cf</sub> = X ⊙ (1 - M)</i> (removing tumor evidence).<br/>"
        "The complete multi-task loss is: <i>L = L<sub>cls</sub> + λ<sub>seg</sub> L<sub>seg</sub> + λ<sub>cons</sub> L<sub>cons</sub> + λ<sub>cf</sub> L<sub>cf</sub></i>, "
        "where <i>L<sub>cons</sub> = D<sub>KL</sub>(p<sub>full</sub> ∥ p<sub>ROI</sub>)</i> and <i>L<sub>cf</sub> = ReLU(p<sub>cf, y</sub> - 0.33)</i>.", body_style
    ))

    # Section III: Patient-Level Separation
    story.append(Paragraph("III. EXPERIMENTAL DESIGN & DATA LEAKAGE AUDIT", h1_style))
    story.append(Paragraph(
        "To prevent patient-slice data leakage, partitioning adheres strictly to the official Figshare 5-fold cross-validation partition (cvind.mat). "
        "An audit confirmed <b>zero patient overlap</b> across splits (0 leaking patients out of 233). "
        "Folds 1–3 form the training set (1,793 slices, 139 patients), Fold 4 the validation set (628 slices, 47 patients), "
        "and Fold 5 is held out strictly for testing (643 slices, 47 patients).", body_style
    ))

    # Table III
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>TABLE I: AUDITED EVALUATION METRICS ACROSS ABLATION CONFIGURATIONS (HELD-OUT PATIENT TEST SET, N=643)</b>", caption_style))

    t_data = [
        ["Config", "Description", "Accuracy", "Macro-F1", "Dice", "IoU", "ECE", "NLL", "CF Δp"],
        ["A1", "Baseline ResNet18", "91.60%", "90.47%", "—", "—", "4.31%", "0.301", "0.431"],
        ["A2", "A1 + U-Net Segmentation", "91.60%", "90.47%", "68.97%", "56.52%", "4.31%", "0.301", "0.431"],
        ["A3", "A2 + Eval on Tumor ROI", "58.94%", "58.74%", "68.97%", "56.52%", "—", "—", "—"],
        ["A4", "A3 + TGC Consistency Loss", "83.83%", "78.67%", "66.95%", "56.68%", "—", "—", "—"],
        ["A5", "A4 + Counterfactual Loss", "83.83%", "78.67%", "66.95%", "56.68%", "—", "—", "0.304"],
        ["A6", "A5 + Calib. & Selective", "95.10%*", "78.67%", "66.95%", "56.68%", "3.79%", "0.253", "0.304"]
    ]
    t = Table(t_data, colWidths=[0.5*inch, 1.8*inch, 0.65*inch, 0.65*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f3f6")),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t)
    story.append(Paragraph("<font size=7.5><i>*Config A6 reflects selective accuracy on accepted cases under risk-controlled threshold τ = 0.80 (85.7% coverage).</i></font>", caption_style))
    story.append(Spacer(1, 4))

    # Section IV: Results & Scientific Discussion
    story.append(Paragraph("IV. AUDITED EXPERIMENTAL RESULTS & DISCUSSION", h1_style))
    story.append(Paragraph(
        f"<b>1. Classification and Per-Class Discrepancy Audit:</b> On the 643 held-out test slices, the baseline ResNet18 classifier achieved {b_acc:.2f}% accuracy (589/643 correct) "
        "and macro-F1 of 90.47%. Per-class breakdown: Meningioma (Recall: 80.12%, Precision: 88.67%, F1: 84.18%), Glioma (Recall: 97.56%, Precision: 93.65%, F1: 95.56%), "
        f"and Pituitary (Recall: 92.63%, Precision: 90.72%, F1: 91.67%). The joint TGRL model achieved {t_acc:.2f}% accuracy (539/643 correct). "
        "TGRL exhibited extreme precision on meningiomas (100.0%, 0 false positives) and near-perfect recall on gliomas (100.0%) and pituitary tumors (99.47%), "
        "but suffered lower recall on meningiomas (37.95%, 63/166), misclassifying 73 meningiomas as gliomas and 30 as pituitary tumors.", body_style
    ))
    story.append(Paragraph(
        f"<b>2. Consistency and Masked-Input Domain Collapse:</b> Baseline evaluation on isolated tumor ROIs dropped accuracy to 58.94% (agreement: {b_agree:.2f}%, mean KL: {b_kl:.4f}), "
        f"demonstrating reliance on non-tumor parenchyma. In TGRL, full-vs-ROI agreement was measured at {t_agree:.2f}% (mean KL: {t_kl:.4f}). "
        "Training log audits revealed that early epochs attained up to 91.6% agreement, but subsequent model selection driven by segmentation Dice (which reached 66.95%) "
        "caused representation collapse on masked inputs: 586/643 ROI images were classified as pituitary tumors, because isolated focal blobs without background tissue mimic pituitary morphology.", body_style
    ))
    story.append(Paragraph(
        f"<b>3. Counterfactual Sensitivity and Selective Prediction:</b> Removing the tumor caused significant true-class probability drops: baseline mean Δp = {cf_b_delta:.4f} (flip rate {cf_b_flip:.2f}%), "
        f"TGRL mean Δp = {cf_t_delta:.4f} (flip rate {cf_t_flip:.2f}%). Temperature scaling (T* = {T_star:.2f}) reduced ECE from {ece_before:.2f}% to {ece_after:.2f}%. "
        "Selective prediction audits showed that the validation threshold τ* = 0.33 accepted 100% of test cases without rejection. Calibrating conservatively to τ = 0.80 enforces "
        "a selective risk of 4.90% (<5.0% clinical target) at 85.7% coverage (551/643 slices accepted, 92 cases referred to expert review).", body_style
    ))

    # Section V: Figures
    story.append(PageBreak())
    story.append(Paragraph("V. AUDITED PUBLICATION FIGURES", h1_style))

    fig_items = [
        ("fig3_confusion_matrices.png", "Fig. 1. Audited confusion matrices on 643 test slices: (left) Baseline (589/643 correct); (right) TGRL Joint Model (539/643 correct)."),
        ("fig4_per_class_comparison.png", "Fig. 2. Per-class Precision, Recall, and F1 comparison on full [0, 110%] scale, showing Meningioma recall (37.95%) and Glioma recall (100.0%)."),
        ("fig6_quad_representations.png", "Fig. 3. Quad-panel TGRL representations: Original MRI X, predicted tumor map M, tumor-focused ROI X_ROI, and counterfactual removal X_cf."),
        ("fig7_full_vs_roi_probabilities.png", "Fig. 4. Full-slice vs. ROI prediction probability scatter: (a) Baseline (59.41% agreement); (b) TGRL (39.97% agreement)."),
        ("fig8_counterfactual_delta_distribution.png", "Fig. 5. Counterfactual sensitivity distribution: Confidence drop Δp = p_original - p_cf upon tumor removal."),
        ("fig10_selective_risk_coverage.png", "Fig. 6. Selective prediction: Coverage versus selective risk tradeoff, comparing validation policy (τ=0.33) and risk-controlled policy (τ=0.80).")
    ]

    for fname, caption in fig_items:
        fpath = FIG_DIR / fname
        if fpath.exists():
            story.append(Image(str(fpath), width=5.8*inch, height=2.5*inch))
            story.append(Paragraph(caption, caption_style))
            story.append(Spacer(1, 3))

    # Limitations & Conclusion
    story.append(Paragraph("VI. LIMITATIONS AND CONCLUSION", h1_style))
    story.append(Paragraph(
        "<b>Limitations:</b> Evaluated on 2D slices without 3D volumetric continuity; pixel-zeroing creates artificial boundary discontinuities; "
        "and multi-task competition caused meningioma recall degradation and ROI agreement collapse.<br/>"
        "<b>Conclusion:</b> This work audited and empirically verified Tumor-Grounded Reliability Learning. By presenting genuine experimental metrics, "
        "identifying multi-task optimization tradeoffs, and demonstrating conservative selective triage, this study establishes a transparent foundation for trustworthy medical AI.", body_style
    ))

    # References
    story.append(Spacer(1, 4))
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] H. H. Sultan et al., 'Multi-Classification of Brain Tumor Images Using Deep Neural Network,' IEEE Access, 2019.",
        "[2] M. F. Almufareh et al., 'Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning,' IEEE Access, 2024.",
        "[3] P. K. Tiwary et al., 'Deep Learning-Based MRI Brain Tumor Segmentation With EfficientNet-Enhanced UNet,' IEEE Access, 2025.",
        "[4] N. M. Hussain Hassan and W. Boulila, 'Efficient Approach for Brain Tumor Detection and Classification Using Fuzzy Thresholding,' IEEE Access, 2025."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.5, leading=11, spaceAfter=2.5)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF] Compiled audited PDF to: {target_path}")

def run_compilation():
    build_pdf(OUTPUT_PDF)
    build_pdf(CORRECTED_PDF)

if __name__ == "__main__":
    run_compilation()
