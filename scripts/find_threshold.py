"""Sweep prediction thresholds on val to find the one that maximizes mIoU.

Run from notebook (or `python -m scripts.find_threshold --ckpt ... --model lss`):
    from scripts.find_threshold import main
    main(ckpt="...best_lss.pt", model_name="lss", tta=True)
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import CKPT_DIR, DATA_ROOT, IMAGE_SIZE
from src.dataset import StaticBEVDataset
from src.factory import build_model
from src.metrics import MeanIoU
from src.submit import _hflip_inputs


@torch.no_grad()
def collect_probs(model, loader, device, tta=False, target_w=IMAGE_SIZE[1]):
    """Returns flat tensors of probs (N,) and gt (N,) over all val pixels (255-masked)."""
    model.eval()
    all_probs, all_gt = [], []
    for batch in tqdm(loader, desc="collect val probs"):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        car2cams = batch["car2cams"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        logits = model(images, intrinsics, car2cams)
        probs = torch.sigmoid(logits)
        if tta:
            images_f, K_f, c2c_f = _hflip_inputs(images, intrinsics, car2cams, target_w)
            logits_f = model(images_f, K_f, c2c_f)
            probs_f = torch.flip(torch.sigmoid(logits_f), dims=[-1])
            probs = (probs + probs_f) / 2

        mask = gt != 255
        all_probs.append(probs[mask].cpu())
        all_gt.append(gt[mask].cpu())
    return torch.cat(all_probs), torch.cat(all_gt)


def sweep_thresholds(probs, gt, thresholds=None):
    if thresholds is None:
        thresholds = [round(0.30 + 0.025 * i, 3) for i in range(17)]  # 0.30..0.70
    rows = []
    for t in thresholds:
        preds = (probs > t).long()
        # Build a 2x2 confusion in one shot
        idx = preds * 2 + gt.long()
        bins = torch.bincount(idx, minlength=4).reshape(2, 2)  # [pred, gt]
        ious = []
        for c in range(2):
            tp = bins[c, c].item()
            fp = bins[c, :].sum().item() - tp
            fn = bins[:, c].sum().item() - tp
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 0.0)
        miou = sum(ious) / 2
        rows.append((t, miou, ious[0], ious[1]))
    return rows


def main(ckpt: str, model_name: str = None, tta: bool = True,
         data_root: str = str(DATA_ROOT), batch_size: int = 8, num_workers: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    name = model_name or (state.get("model_name") if isinstance(state, dict) else None) or "lss"
    model = build_model(name).to(device)
    model.load_state_dict(sd)
    print(f"Loaded {name} from {ckpt}")

    val_ds = StaticBEVDataset(data_root, split="val")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    probs, gt = collect_probs(model, val_loader, device, tta=tta)
    rows = sweep_thresholds(probs, gt)
    print(f"\n{'thr':>6} {'mIoU':>8} {'free':>8} {'occ':>8}")
    best = max(rows, key=lambda r: r[1])
    for t, miou, fr, oc in rows:
        marker = "  ←" if (t, miou) == (best[0], best[1]) else ""
        print(f"{t:6.3f} {miou:8.4f} {fr:8.4f} {oc:8.4f}{marker}")
    print(f"\nBest threshold: {best[0]:.3f} (mIoU={best[1]:.4f})")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no_tta", dest="tta", action="store_false")
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    main(args.ckpt, args.model, args.tta, args.data_root, args.batch_size, args.num_workers)
