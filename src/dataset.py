"""Dataset for the static obstacle BEV contest.

info.csv stores paths relative to ``data_root`` (the parent of the
``autonomy_yandex_dataset_*`` folders). We resolve them against ``data_root``.
"""
import random
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


def build_image_transform(image_size=IMAGE_SIZE, training=False):
    transforms = [v2.PILToTensor()]
    if training:
        transforms.append(
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03)
        )
    transforms += [
        v2.Resize(image_size, antialias=True),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return v2.Compose(transforms)


class StaticBEVDataset(Dataset):
    """Returns dict with images, intrinsics, car2cams, gt (train/val), index.

    Intrinsics are scaled to match the resized image size — original images are
    1024 wide and ~540-570 tall, but cameras have different heights so scale
    must be computed per camera.
    """

    def __init__(self, data_root=DATA_ROOT, split="train", transform=None,
                 target_size=IMAGE_SIZE, hflip_prob=0.0):
        assert split in SPLIT_DIRS, f"unknown split: {split}"
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / SPLIT_DIRS[split]
        self.info = pd.read_csv(self.split_dir / "info.csv", index_col=0)
        self.transform = transform or build_image_transform(
            target_size, training=(split == "train")
        )
        self.target_h, self.target_w = target_size
        self.hflip_prob = hflip_prob if split == "train" else 0.0

    def __len__(self):
        return len(self.info)

    def _resolve(self, rel_path):
        return self.data_root / rel_path

    def __getitem__(self, idx):
        row = self.info.iloc[idx]

        images_pil = [
            Image.open(self._resolve(row[name])).convert("RGB")
            for name in CAMERA_NAMES
        ]
        orig_sizes = [img.size for img in images_pil]  # list of (W, H)
        images = torch.stack([self.transform(img) for img in images_pil])  # (N, 3, h, w)

        intrinsics_list = []
        for name, (orig_w, orig_h) in zip(INTRINSICS_NAMES, orig_sizes):
            K = np.load(self._resolve(row[name])).copy().astype(np.float32)  # (3, 4)
            sx = self.target_w / orig_w
            sy = self.target_h / orig_h
            K[0, :] *= sx   # scales fx, skew (=0), cx, last col (=0)
            K[1, :] *= sy   # scales fy, cy, last col (=0)
            intrinsics_list.append(K)
        intrinsics = np.stack(intrinsics_list)
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

        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            sample = self._apply_hflip(sample)

        return sample

    def _apply_hflip(self, sample):
        """Horizontally flip images and update K, car_to_cam, GT to stay consistent.

        Image flip changes principal point cx → (W-1)-cx in the resized intrinsic.
        World-Y is mirrored in car frame; cam-X is mirrored in camera frame.
        car_to_cam_aug = M_x @ car_to_cam_orig @ M_y, with
            M_x = diag(-1, 1, 1, 1) (cam-frame X-mirror),
            M_y = diag(1, -1, 1, 1) (car-frame Y-mirror).
        """
        sample["images"] = torch.flip(sample["images"], dims=[-1])

        K = sample["intrinsics"].clone()
        K[..., 0, 2] = (self.target_w - 1) - K[..., 0, 2]
        sample["intrinsics"] = K

        c2c = sample["car2cams"]
        M_x = torch.eye(4, dtype=c2c.dtype)
        M_x[0, 0] = -1
        M_y = torch.eye(4, dtype=c2c.dtype)
        M_y[1, 1] = -1
        sample["car2cams"] = M_x @ c2c @ M_y

        if "gt" in sample:
            sample["gt"] = torch.flip(sample["gt"], dims=[-1])

        return sample
