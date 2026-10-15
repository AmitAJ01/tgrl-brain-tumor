"""
Pre-processing and caching script for Figshare CE-MRI dataset.
Reads raw .mat files according to results/split_manifest.json,
normalizes, resizes to 224x224, and stores in data/cache/ for ultra-fast CPU training.
"""

import os
import json
import h5py
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import normalize_intensity, resize_pair


RAW_DIR = Path(r"C:\Users\amitr\Desktop\brain-tumor-ai-proj\data\raw")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "results" / "split_manifest.json"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_cache():
    print(f"[CACHE] Loading split manifest from: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for split in ["train", "val", "test"]:
        records = manifest[split]
        print(f"[CACHE] Processing {split} split ({len(records)} slices)...")
        
        imgs_list = []
        masks_list = []
        labels_list = []
        pids_list = []
        slice_ids_list = []

        for i, rec in enumerate(records):
            fpath = RAW_DIR / rec["filename"]
            with h5py.File(fpath, "r") as f:
                cj = f["cjdata"]
                img_raw = np.array(cj["image"]).T
                mask_raw = np.array(cj["tumorMask"]).T

            # Normalize raw float intensities to [0, 1]
            img_norm = normalize_intensity(img_raw)
            # Resize image and mask
            img_resized, mask_resized = resize_pair(img_norm, mask_raw, target_size=(224, 224))
            
            # Store image as uint8 [0, 255] to save memory
            imgs_list.append((img_resized * 255.0).astype(np.uint8))
            masks_list.append(mask_resized.astype(np.uint8))
            labels_list.append(rec["class_idx"])
            pids_list.append(rec["pid"])
            slice_ids_list.append(rec["slice_id"])

        out_path = CACHE_DIR / f"{split}.npz"
        np.savez_compressed(
            out_path,
            images=np.stack(imgs_list, axis=0),
            masks=np.stack(masks_list, axis=0),
            labels=np.array(labels_list, dtype=np.int64),
            pids=np.array(pids_list),
            slice_ids=np.array(slice_ids_list, dtype=np.int64)
        )
        print(f"[CACHE] Saved {split}.npz ({os.path.getsize(out_path) / (1024**2):.2f} MB)")

    print("[CACHE] Dataset caching complete!")

if __name__ == "__main__":
    build_cache()
