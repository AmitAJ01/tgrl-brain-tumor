"""
PyTorch Dataset and DataLoader implementation for TGRL.
Loads standardized patient-split data from cached npz archives.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import JointTransform

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]

class TGRLDataset(Dataset):
    def __init__(self, npz_path: str, is_train: bool = False):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=True)
        self.images = data["images"]  # [N, 224, 224] uint8
        self.masks = data["masks"]    # [N, 224, 224] uint8 {0, 1}
        self.labels = data["labels"]  # [N] int64
        self.pids = data["pids"]      # [N] str
        self.slice_ids = data["slice_ids"]  # [N] int64
        self.is_train = is_train
        self.transform = JointTransform(is_train=is_train)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Image: [224, 224] in [0, 255] uint8 -> float32 in [0, 1]
        img_np = (self.images[idx].astype(np.float32) / 255.0)
        # Repeat single-channel grayscale to 3 channels for backbone compatibility
        img_3ch = np.repeat(img_np[np.newaxis, :, :], 3, axis=0) # [3, 224, 224]
        img_tensor = torch.from_numpy(img_3ch).float()
        
        # Mask: [224, 224] in {0, 1} uint8 -> float32 [1, 224, 224]
        mask_np = self.masks[idx].astype(np.float32)[np.newaxis, :, :]
        mask_tensor = torch.from_numpy(mask_np).float()

        # Keep un-normalized [0, 1] copy for ROI masking: X_ROI = X ⊙ M
        raw_img_tensor = img_tensor.clone()

        # Apply joint spatial augmentations (flips, rotations) and ImageNet normalization
        norm_img_tensor, aug_mask_tensor = self.transform(img_tensor, mask_tensor)

        return {
            "image": norm_img_tensor,          # Normalized for classification [3, 224, 224]
            "raw_image": raw_img_tensor,      # Min-max in [0, 1] for ROI calculation [3, 224, 224]
            "mask": aug_mask_tensor,          # Binary ground truth mask [1, 224, 224]
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "pid": str(self.pids[idx]),
            "slice_id": int(self.slice_ids[idx])
        }

def get_dataloaders(cache_dir: str = "data/cache", batch_size: int = 32, num_workers: int = 0):
    cache_path = Path(cache_dir)
    train_ds = TGRLDataset(cache_path / "train.npz", is_train=True)
    val_ds = TGRLDataset(cache_path / "val.npz", is_train=False)
    test_ds = TGRLDataset(cache_path / "test.npz", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
