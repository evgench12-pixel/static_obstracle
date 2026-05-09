"""Shared constants and default training config."""
from pathlib import Path

DATA_ROOT = Path.home() / "static_obstracle" / "data"
CKPT_DIR = Path.home() / "static_obstracle" / "checkpoints"
SUBMISSION_DIR = Path.home() / "static_obstracle" / "submissions"

CAMERA_NAMES = [
    "/camera/inner/frontal/middle",
    "/camera/inner/frontal/far",
    "/side/left/forward",
    "/side/right/forward",
]
INTRINSICS_NAMES = [f"{c}/intrinsic_params" for c in CAMERA_NAMES]
CAR2CAM_NAMES = [f"{c}/car_to_cam" for c in CAMERA_NAMES]

OUT_SIZE = (188, 126)
IMAGE_SIZE = (256, 512)
IGNORE_INDEX = 255

SPLIT_DIRS = {
    "train": "autonomy_yandex_dataset_train",
    "val": "autonomy_yandex_dataset_val",
    "test": "autonomy_yandex_dataset_test",
}

TRAIN_CONFIG = {
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "log_every": 50,
    "amp": True,
}
