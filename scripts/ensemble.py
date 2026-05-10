"""Ensemble multiple LSS checkpoints — average sigmoid probs (each with TTA).

Sweeps thresholds on val to pick the optimum, then generates the test
submission zip.

Usage from a notebook:
    from scripts.ensemble import main
    main(
        checkpoints=[
            ("~/static_obstracle/checkpoints/best_lss.pt", "lss"),
            ("~/static_obstracle/checkpoints/best_lss_r50.pt", "lss_r50"),
        ],
        out="~/static_obstracle/submissions/submission_ensemble.zip",
    )
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import DATA_ROOT, IMAGE_SIZE, SPLIT_DIRS
from src.dataset import StaticBEVDataset
from src.factory import build_model
from src.submit import _hflip_inputs, build_zip


def _expand(p):
    return Path(p).expanduser()


def load_models(checkpoints, device):
    models = []
    for ckpt_path, model_name in checkpoints:
        ckpt_path = _expand(ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        name = (model_name
                or (state.get("model_name") if isinstance(state, dict) else None)
                or "lss")
        m = build_model(name).to(device).eval()
        m.load_state_dict(sd)
        models.append((name, m))
        print(f"Loaded {name} from {ckpt_path}")
    return models


@torch.no_grad()
def _ensemble_probs(models, images, intrinsics, car2cams, tta, target_w):
    probs_sum = None
    for _, m in models:
        logits = m(images, intrinsics, car2cams)
        p = torch.sigmoid(logits)
        if tta:
            i_f, K_f, c_f = _hflip_inputs(images, intrinsics, car2cams, target_w)
            logits_f = m(i_f, K_f, c_f)
            p_f = torch.flip(torch.sigmoid(logits_f), dims=[-1])
            p = (p + p_f) / 2
        probs_sum = p if probs_sum is None else probs_sum + p
    return probs_sum / len(models)


@torch.no_grad()
def collect_val_probs(models, data_root, batch_size, num_workers, device, tta):
    val_ds = StaticBEVDataset(data_root, split="val")
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    target_w = IMAGE_SIZE[1]
    all_probs, all_gt = [], []
    for batch in tqdm(loader, desc="ensemble val"):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        car2cams = batch["car2cams"].to(device, non_blocking=True)
        gt = batch["gt"].to(device)
        probs = _ensemble_probs(models, images, intrinsics, car2cams, tta, target_w)
        mask = gt != 255
        all_probs.append(probs[mask].cpu())
        all_gt.append(gt[mask].cpu())
    return torch.cat(all_probs), torch.cat(all_gt)


def sweep_thresholds(probs, gt, thresholds=None):
    if thresholds is None:
        thresholds = [round(0.30 + 0.025 * i, 3) for i in range(17)]
    rows = []
    for t in thresholds:
        preds = (probs > t).long()
        idx = preds * 2 + gt.long()
        bins = torch.bincount(idx, minlength=4).reshape(2, 2)
        ious = []
        for c in range(2):
            tp = bins[c, c].item()
            fp = bins[c, :].sum().item() - tp
            fn = bins[:, c].sum().item() - tp
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 0.0)
        rows.append((t, sum(ious) / 2, ious[0], ious[1]))
    return rows


@torch.no_grad()
def predict_test(models, data_root, batch_size, num_workers, device, tta, threshold):
    test_dir = Path(data_root) / SPLIT_DIRS["test"]
    info = pd.read_csv(test_dir / "info.csv", index_col=0)
    ds = StaticBEVDataset(data_root, split="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    pred_dir = test_dir / "predicted_static_grids"
    pred_dir.mkdir(exist_ok=True)
    target_w = IMAGE_SIZE[1]
    written = 0
    for batch in tqdm(loader, desc="ensemble test"):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        car2cams = batch["car2cams"].to(device, non_blocking=True)
        idxs = batch["index"].tolist()
        probs = _ensemble_probs(models, images, intrinsics, car2cams, tta, target_w)
        preds = (probs > threshold).cpu().numpy().astype(np.int32)
        for i, idx in enumerate(idxs):
            rel = info.iloc[idx]["predicted_occupancy_grid"]
            out_path = Path(data_root) / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, preds[i])
            written += 1
    print(f"Wrote {written} predictions")
    return test_dir


def main(checkpoints, out, data_root=str(DATA_ROOT), batch_size=4, num_workers=4,
         tta=True, threshold=None):
    """checkpoints: list of (ckpt_path, model_name) tuples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(checkpoints, device)

    if threshold is None:
        probs, gt = collect_val_probs(models, data_root, batch_size, num_workers, device, tta)
        rows = sweep_thresholds(probs, gt)
        best = max(rows, key=lambda r: r[1])
        print(f"\n{'thr':>6} {'mIoU':>8}")
        for t, miou, _, _ in rows:
            mark = "  ←" if t == best[0] else ""
            print(f"{t:6.3f} {miou:8.4f}{mark}")
        threshold = best[0]
        print(f"Best threshold: {threshold:.3f} (val mIoU={best[1]:.4f})")

    test_dir = predict_test(models, data_root, batch_size, num_workers, device,
                            tta, threshold)
    build_zip(test_dir, _expand(out))
