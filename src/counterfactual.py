"""
Counterfactual Tumor Removal Analysis (Component F & Evaluation Improvements).
Replaces simple black-masking with:
  1. Generative in-painting (Telea / Navier-Stokes / PDE biharmonic inpainting).
  2. Matched normal-tissue noise controls (synthesizing surrounding parenchyma texture).
This isolates genuine pathological reliance from synthetic zero-masking boundary artifacts.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation
from src.metrics import compute_counterfactual_metrics

def inpaint_slice_cv2(image_np: np.ndarray, mask_np: np.ndarray, inpaint_radius: int = 5) -> np.ndarray:
    """
    Inpaints the tumor region of a single image using Telea / Navier-Stokes PDE inpainting.
    image_np: [H, W, C] in [0, 1] float
    mask_np:  [H, W] binary uint8 {0, 255}
    """
    # Scale to 0-255 uint8 for OpenCV
    img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255

    inpainted_channels = []
    for c in range(image_np.shape[2]):
        ch = img_uint8[:, :, c]
        ch_inpaint = cv2.inpaint(ch, mask_uint8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        inpainted_channels.append(ch_inpaint.astype(np.float32) / 255.0)

    inpainted = np.stack(inpainted_channels, axis=2)

    # Smooth boundary blending with normalized mask [0, 1]
    mask_float = (mask_np > 0).astype(np.float32)
    smooth_mask = gaussian_filter(mask_float, sigma=2.0)[:, :, None]
    smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
    blended = (1.0 - smooth_mask) * image_np + smooth_mask * inpainted
    return np.clip(blended, 0.0, 1.0)

def matched_noise_slice(image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Fills the tumor region with Gaussian noise matching the mean and variance
    of the surrounding healthy brain parenchyma (extracted via dilation ring).
    """
    bin_mask = mask_np > 0.5
    # Dilation ring of 7 pixels around tumor
    dilated = binary_dilation(bin_mask, iterations=7)
    ring = dilated & (~bin_mask)

    result = image_np.copy()
    mask_float = (mask_np > 0).astype(np.float32)
    smooth_mask = gaussian_filter(mask_float, sigma=2.0)
    smooth_mask = np.clip(smooth_mask, 0.0, 1.0)

    for c in range(image_np.shape[2]):
        ch = image_np[:, :, c]
        if np.any(ring):
            mu = float(np.mean(ch[ring]))
            sigma = float(np.std(ch[ring]))
            sigma = max(sigma, 0.02)
        else:
            mu = 0.45
            sigma = 0.08

        # Synthesize correlated texture noise matching parenchyma
        noise = np.random.normal(loc=mu, scale=sigma, size=ch.shape)
        noise = gaussian_filter(noise, sigma=1.0)

        ch_blended = (1.0 - smooth_mask) * ch + smooth_mask * noise
        result[:, :, c] = ch_blended

    return np.clip(result, 0.0, 1.0)


def generate_counterfactual_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    method: str = "inpaint",
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Generates counterfactual images:
      - method="inpaint": Generative tissue inpainting.
      - method="matched_noise": Tissue-matched parenchyma noise controls.
      - method="black_mask": Legacy zero-fill mask X ⊙ (1 - M).
    images: [B, 3, H, W] in [0, 1]
    masks:  [B, 1, H, W] probability maps
    """
    if method == "black_mask":
        bin_masks = (masks >= threshold).float()
        return images * (1.0 - bin_masks)

    # Process slices with generative inpainting or matched noise
    b, c, h, w = images.shape
    device = images.device
    imgs_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
    masks_np = (masks.detach().cpu() >= threshold).squeeze(1).numpy().astype(np.uint8)  # [B, H, W]

    cf_batch = np.zeros_like(imgs_np)
    for i in range(b):
        img_i = imgs_np[i]
        mask_i = masks_np[i]

        if not np.any(mask_i):
            cf_batch[i] = img_i
            continue

        if method == "inpaint":
            cf_batch[i] = inpaint_slice_cv2(img_i, mask_i)
        elif method == "matched_noise":
            cf_batch[i] = matched_noise_slice(img_i, mask_i)
        else:
            cf_batch[i] = img_i * (1.0 - mask_i[:, :, None])

    cf_tensor = torch.from_numpy(cf_batch).permute(0, 3, 1, 2).float().to(device)
    return cf_tensor

def run_counterfactual_analysis(
    classifier,
    test_loader,
    device="cpu",
    method="inpaint",
    mask_threshold: float = 0.5,
    seg_model=None
):
    """
    Runs counterfactual analysis across test dataset using the specified counterfactual method.
    """
    classifier.eval()
    if seg_model is not None:
        seg_model.eval()

    all_p_full = []
    all_p_cf = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            target = batch["label"].to(device)

            if seg_model is not None:
                mask = seg_model(x)
            else:
                mask = batch["mask"].to(device)

            # Predictions on original image
            if hasattr(classifier, "classify_image"):
                logits_full = classifier.classify_image(x)
            else:
                logits_full = classifier(x)
            p_full = torch.softmax(logits_full, dim=1)

            # Counterfactual image using specified control method
            x_cf = generate_counterfactual_batch(x, mask, method=method, threshold=mask_threshold)
            if hasattr(classifier, "classify_image"):
                logits_cf = classifier.classify_image(x_cf)
            else:
                logits_cf = classifier(x_cf)
            p_cf = torch.softmax(logits_cf, dim=1)

            all_p_full.append(p_full.cpu().numpy())
            all_p_cf.append(p_cf.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_p_full = np.concatenate(all_p_full, axis=0)
    all_p_cf = np.concatenate(all_p_cf, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    summary_metrics = compute_counterfactual_metrics(all_p_full, all_p_cf, all_targets)

    y_pred_full = np.argmax(all_p_full, axis=1)
    y_pred_cf = np.argmax(all_p_cf, axis=1)
    delta_true = all_p_full[np.arange(len(all_targets)), all_targets] - all_p_cf[np.arange(len(all_targets)), all_targets]
    delta_pred = all_p_full[np.arange(len(all_targets)), y_pred_full] - all_p_cf[np.arange(len(all_targets)), y_pred_full]

    return {
        "method": method,
        "summary": summary_metrics,
        "p_full": all_p_full,
        "p_cf": all_p_cf,
        "y_true": all_targets,
        "y_pred_full": y_pred_full,
        "y_pred_cf": y_pred_cf,
        "delta_true": delta_true,
        "delta_pred": delta_pred
    }

def run_counterfactual_control_study(classifier, test_loader, device="cpu", seg_model=None):
    """
    Executes all three counterfactual controls (Black Mask, Generative Inpaint, Matched Noise)
    to isolate genuine pathological reliance from synthetic masking artifacts.
    """
    res_black = run_counterfactual_analysis(classifier, test_loader, device, method="black_mask", seg_model=seg_model)
    res_inpaint = run_counterfactual_analysis(classifier, test_loader, device, method="inpaint", seg_model=seg_model)
    res_noise = run_counterfactual_analysis(classifier, test_loader, device, method="matched_noise", seg_model=seg_model)

    delta_black = res_black["summary"]["all_cases"]["mean_delta_true_class"]
    delta_inpaint = res_inpaint["summary"]["all_cases"]["mean_delta_true_class"]
    delta_noise = res_noise["summary"]["all_cases"]["mean_delta_true_class"]

    genuine_ratio = delta_inpaint / max(delta_black, 1e-6)
    synthetic_artifact_ratio = 1.0 - genuine_ratio

    return {
        "black_mask": res_black["summary"],
        "generative_inpaint": res_inpaint["summary"],
        "matched_noise": res_noise["summary"],
        "mean_delta_black": float(delta_black),
        "mean_delta_inpaint": float(delta_inpaint),
        "mean_delta_noise": float(delta_noise),
        "genuine_pathological_reliance_pct": float(genuine_ratio * 100.0),
        "synthetic_artifact_fraction_pct": float(synthetic_artifact_ratio * 100.0)
    }
