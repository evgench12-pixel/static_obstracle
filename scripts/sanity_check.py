"""Smoke test: load one batch, run model, compute loss + metric, validate shapes.

Run from the project root:
    python -m scripts.sanity_check                # default (multicam_cnn)
    python -m scripts.sanity_check --model lss    # LSS

Or from a notebook:
    from scripts.sanity_check import main
    main(model_name="lss")
"""
import argparse

import torch
from torch.utils.data import DataLoader

from src.config import OUT_SIZE
from src.dataset import StaticBEVDataset
from src.factory import build_model
from src.losses import masked_bce_with_logits
from src.metrics import MeanIoU


def main(model_name: str = "multicam_cnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")

    ds = StaticBEVDataset(split="val")
    print(f"Val samples: {len(ds)}")

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else v}")

    model = build_model(model_name).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    images = batch["images"].to(device)
    intrinsics = batch["intrinsics"].to(device)
    car2cams = batch["car2cams"].to(device)
    gt = batch["gt"].to(device)

    logits = model(images, intrinsics, car2cams)
    print(f"Logits shape: {tuple(logits.shape)}, expected (B, 1, {OUT_SIZE[0]}, {OUT_SIZE[1]})")
    assert logits.shape[1:] == (1, OUT_SIZE[0], OUT_SIZE[1]), "output shape mismatch"

    loss = masked_bce_with_logits(logits, gt)
    print(f"Loss: {loss.item():.4f}")

    metric = MeanIoU()
    preds = (torch.sigmoid(logits) > 0.5).long()
    metric.update(preds, gt)
    print(f"Metric on this batch: {metric.compute()}")

    print("\nSanity check passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="multicam_cnn", choices=["multicam_cnn", "lss"])
    args = p.parse_args()
    main(args.model)
