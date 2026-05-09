"""Generate test predictions and build a submission zip.

Usage: python -m src.submit --ckpt ~/static_obstracle/checkpoints/best.pt
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import (
    CKPT_DIR,
    DATA_ROOT,
    SPLIT_DIRS,
    SUBMISSION_DIR,
    TRAIN_CONFIG,
)
from src.dataset import StaticBEVDataset
from src.model import MultiCamCNN


@torch.no_grad()
def predict_test(model, data_root, batch_size, num_workers, device):
    test_dir = Path(data_root) / SPLIT_DIRS["test"]
    info = pd.read_csv(test_dir / "info.csv", index_col=0)
    ds = StaticBEVDataset(data_root, split="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pred_dir = test_dir / "predicted_static_grids"
    pred_dir.mkdir(exist_ok=True)

    model.eval()
    written = 0
    for batch in tqdm(loader, desc="test inference"):
        images = batch["images"].to(device, non_blocking=True)
        idxs = batch["index"].tolist()
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)  # (B, 1, 188, 126)
        for i, idx in enumerate(idxs):
            rel = info.iloc[idx]["predicted_occupancy_grid"]
            out_path = Path(data_root) / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, preds[i])
            written += 1
    print(f"Wrote {written} prediction files to {pred_dir}")
    return test_dir


def build_zip(test_dir, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(test_dir / "info.csv", arcname="info.csv")
        pred_dir = test_dir / "predicted_static_grids"
        for npy in sorted(pred_dir.glob("*.npy")):
            zf.write(npy, arcname=f"predicted_static_grids/{npy.name}")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"Built {out_path} ({size_mb:.2f} MB)")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiCamCNN().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {args.ckpt}")

    test_dir = predict_test(model, args.data_root, args.batch_size, args.num_workers, device)
    build_zip(test_dir, Path(args.out))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(Path(CKPT_DIR) / "best.pt"))
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--out", type=str, default=str(SUBMISSION_DIR / "submission.zip"))
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--num_workers", type=int, default=TRAIN_CONFIG["num_workers"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
