"""Dataset for the static obstacle BEV contest.

info.csv stores paths relative to ``data_root`` (the parent of the
``autonomy_yandex_dataset_*`` folders). We resolve them against ``data_root``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.config import (
    CAMERA_NAMES,
    CAR2CAM_NAMES,
    DATA_ROOT,
    IMAGE_SIZE,
    INTRINSICS_NAMES,
    SPLIT_DIRS,
)


def build_image_transform(image_size=IMAGE_SIZE):
    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize(image_size, antialias=True),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class StaticBEVDataset(Dataset):
    """Returns dict with images, intrinsics, car2cams, gt (train/val), index."""

    def __init__(self, data_root=DATA_ROOT, split="train", transform=None):
        assert split in SPLIT_DIRS, f"unknown split: {split}"
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / SPLIT_DIRS[split]
        self.info = pd.read_csv(self.split_dir / "info.csv", index_col=0)
        self.transform = transform or build_image_transform()

    def __len__(self):
        return len(self.info)

    def _resolve(self, rel_path):
        return self.data_root / rel_path

    def __getitem__(self, idx):
        row = self.info.iloc[idx]

        images = torch.stack([
            self.transform(Image.open(self._resolve(row[name])).convert("RGB"))
            for name in CAMERA_NAMES
        ])  # (N, 3, H, W)

        intrinsics = np.stack([np.load(self._resolve(row[name])) for name in INTRINSICS_NAMES])
        car2cams = np.stack([np.load(self._resolve(row[name])) for name in CAR2CAM_NAMES])

        sample = {
            "images": images,
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "car2cams": torch.from_numpy(car2cams).float(),
            "index": idx,
        }
        if self.split != "test":
            gt = np.load(self._resolve(row["gt_occupancy_grid"]))  # (1, 188, 126), int32
            sample["gt"] = torch.from_numpy(gt).long()
        return sample
