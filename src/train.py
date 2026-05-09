"""Phase A training loop. Run from project root: python -m src.train"""
import argparse
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import CKPT_DIR, DATA_ROOT, TRAIN_CONFIG
from src.dataset import StaticBEVDataset
from src.factory import build_model
from src.losses import masked_bce_with_logits
from src.metrics import MeanIoU


def train_one_epoch(model, loader, optim, scaler, device, log_every=50, pos_weight=None):
    model.train()
    losses = []
    pbar = tqdm(loader, desc="train")
    for i, batch in enumerate(pbar):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        car2cams = batch["car2cams"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            logits = model(images, intrinsics, car2cams)
            loss = masked_bce_with_logits(logits, gt, pos_weight=pos_weight)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        losses.append(loss.item())
        if i % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanIoU()
    for batch in tqdm(loader, desc="eval"):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        car2cams = batch["car2cams"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        logits = model(images, intrinsics, car2cams)
        preds = (torch.sigmoid(logits) > 0.5).long()
        metric.update(preds, gt)
    return metric.compute()


def main(args):
    cfg = {**TRAIN_CONFIG, **vars(args)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    train_ds = StaticBEVDataset(cfg["data_root"], split="train", hflip_prob=cfg["hflip_prob"])
    val_ds = StaticBEVDataset(cfg["data_root"], split="val")
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    model = build_model(cfg["model"]).to(device)
    print(f"Model: {cfg['model']} ({sum(p.numel() for p in model.parameters()):,} params)")
    optim = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optim, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    scaler = GradScaler() if (cfg["amp"] and device.type == "cuda") else None

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optim, scaler, device, cfg["log_every"],
            pos_weight=cfg["pos_weight"],
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | "
            f"val_mIoU={val_metrics['mIoU']:.4f} "
            f"(free={val_metrics['IoU_free']:.4f}, occ={val_metrics['IoU_occupied']:.4f}) | "
            f"lr={optim.param_groups[0]['lr']:.2e} | time={elapsed:.1f}s"
        )
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            ckpt_path = ckpt_dir / f"best_{cfg['model']}.pt"
            torch.save({
                "model": model.state_dict(),
                "model_name": cfg["model"],
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  saved best (mIoU={best_miou:.4f}) → {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="multicam_cnn",
                   choices=["multicam_cnn", "lss", "lss_r50"])
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--ckpt_dir", type=str, default=str(CKPT_DIR))
    p.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--num_workers", type=int, default=TRAIN_CONFIG["num_workers"])
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    p.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    p.add_argument("--log_every", type=int, default=TRAIN_CONFIG["log_every"])
    p.add_argument("--amp", action="store_true", default=TRAIN_CONFIG["amp"])
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--hflip_prob", type=float, default=0.5,
                   help="probability of horizontal flip augmentation (train split only)")
    p.add_argument("--pos_weight", type=float, default=1.5,
                   help="BCE pos_weight on the occupied class (1.0 = balanced)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
