"""Smoke test: load one batch, run model, compute loss + metric, validate shapes.

Run from the project root: python -m scripts.sanity_check
"""
import torch
from torch.utils.data import DataLoader

from src.config import OUT_SIZE
from src.dataset import StaticBEVDataset
from src.losses import masked_bce_with_logits
from src.metrics import MeanIoU
from src.model import MultiCamCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = StaticBEVDataset(split="val")
    print(f"Val samples: {len(ds)}")

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else v}")

    model = MultiCamCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    images = batch["images"].to(device)
    gt = batch["gt"].to(device)

    logits = model(images)
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
    main()
