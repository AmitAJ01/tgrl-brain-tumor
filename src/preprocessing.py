"""
MRI Preprocessing and Joint Image-Mask Augmentation for TGRL.
Preserves strict spatial alignment between MRI slice and tumor segmentation mask.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
import random

# ImageNet standard normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """
    Min-max normalize raw 16-bit MRI intensities to [0.0, 1.0] float32.
    Safeguards against zero-variance slices.
    """
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        norm = (img - img_min) / (img_max - img_min)
    else:
        norm = np.zeros_like(img, dtype=np.float32)
    return norm

def resize_pair(image: np.ndarray, mask: np.ndarray, target_size=(224, 224)) -> tuple:
    """
    Resize image and mask together.
    Uses bilinear interpolation for continuous MRI intensity and nearest neighbor for binary mask.
    """
    img_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return img_resized, (mask_resized > 0).astype(np.uint8)

class JointTransform:
    """
    Applies identical spatial transformations to image and mask during training.
    Prevents spatial misalignment between tumor anatomy and segmentation targets.
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """
        img_tensor: [3, H, W] in [0, 1]
        mask_tensor: [1, H, W] binary {0, 1}
        """
        if self.is_train:
            # Random horizontal flip (p=0.5)
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # Random gentle rotation (-15 to 15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                img_tensor = TF.rotate(img_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Apply ImageNet normalization to image
        norm_img = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return norm_img, mask_tensor
