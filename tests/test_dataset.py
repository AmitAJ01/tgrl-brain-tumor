"""
Unit tests for TGRL Dataset and Preprocessing.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import TGRLDataset, get_dataloaders
from src.preprocessing import normalize_intensity, resize_pair, JointTransform

def test_intensity_normalization():
    # Constant image
    img_const = np.ones((100, 100), dtype=np.float32) * 500.0
    norm_const = normalize_intensity(img_const)
    assert norm_const.shape == (100, 100)
    assert np.all(norm_const == 0.0)

    # Gradient image
    img_grad = np.arange(0, 1000, dtype=np.float32).reshape(20, 50)
    norm_grad = normalize_intensity(img_grad)
    assert abs(norm_grad.min() - 0.0) < 1e-5
    assert abs(norm_grad.max() - 1.0) < 1e-5

def test_resize_pair():
    img = np.random.rand(512, 512).astype(np.float32)
    mask = (np.random.rand(512, 512) > 0.5).astype(np.uint8)

    img_res, mask_res = resize_pair(img, mask, (224, 224))
    assert img_res.shape == (224, 224)
    assert mask_res.shape == (224, 224)
    assert set(np.unique(mask_res)).issubset({0, 1})

def test_joint_transform():
    transform = JointTransform(is_train=True)
    img_t = torch.rand(3, 224, 224)
    mask_t = (torch.rand(1, 224, 224) > 0.5).float()

    norm_img, aug_mask = transform(img_t, mask_t)
    assert norm_img.shape == (3, 224, 224)
    assert aug_mask.shape == (1, 224, 224)
    assert set(torch.unique(aug_mask).tolist()).issubset({0.0, 1.0})

def test_dataset_loader():
    cache_path = PROJECT_ROOT / "data" / "cache"
    if (cache_path / "val.npz").exists():
        ds = TGRLDataset(cache_path / "val.npz", is_train=False)
        assert len(ds) == 628
        item = ds[0]
        assert "image" in item and "mask" in item and "label" in item
        assert item["image"].shape == (3, 224, 224)
        assert item["mask"].shape == (1, 224, 224)

if __name__ == "__main__":
    test_intensity_normalization()
    test_resize_pair()
    test_joint_transform()
    test_dataset_loader()
    print("All dataset tests passed!")
