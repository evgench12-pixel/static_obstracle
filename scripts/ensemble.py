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
    """checkpoints: list of dicts/tuples. Each may specify image_size override."""
    models = []
    for entry in checkpoints:
        if isinstance(entry, dict):
            ckpt_path, model_name = entry["ckpt"], entry.get("model")
            override_size = entry.get("image_size")
        else:
            ckpt_path, model_name = entry[0], entry[1] if len(entry) > 1 else None
            override_size = entry[2] if len(entry) > 2 else None
        ckpt_path = _expand(ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        name = (model_name
                or (state.get("model_name") if isinstance(state, dict) else None)
                or "lss")
        if override_size is not None:
            image_size = tuple(override_size)
        elif isinstance(state, dict) and state.get("image_size") is not None:
            image_size = tuple(state["image_size"])
        else:
            image_size = IMAGE_SIZE

        kw = {"image_size": image_size} if name.startswith("lss") else {}
        m = build_model(name, **kw).to(device).eval()
        m.load_state_dict(sd)
        models.append({"name": name, "model": m, "image_size": image_size})
        print(f"Loaded {name} (image_size={image_size}) from {ckpt_path}")
    return models


@torch.no_grad()
def _model_probs(model_entry, batch, device, tta):
    """Run a single model with optional TTA at its native image_size."""
    m = model_entry["model"]
    target_w = model_entry["image_size"][1]
    images = batch["images"].to(device, non_blocking=True)
    intrinsics = batch["intrinsics"].to(device, non_blocking=True)
    car2cams = batch["car2cams"].to(device, non_blocking=True)

    logits = m(images, intrinsics, car2cams)
    p = torch.sigmoid(logits)
    if tta:
        i_f, K_f, c_f = _hflip_inputs(images, intrinsics, car2cams, target_w)
        logits_f = m(i_f, K_f, c_f)
        p_f = torch.flip(torch.sigmoid(logits_f), dims=[-1])
        p = (p + p_f) / 2
    return p


@torch.no_grad()
def _ensemble_probs(models, images, intrinsics, car2cams, tta, target_w):
    """Backwards-compat shim: same image_size for all models (used by gen_pseudo)."""
    probs_sum = None
    for entry in models:
        m = entry["model"] if isinstance(entry, dict) else entry[1]
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
    """Each model runs over val at its own image_size; we average resulting BEV probs.

    We accumulate per-pixel sums across models; outputs are always (1, 188, 126),
    so we just sum and divide regardless of the input image size.
    """
    n_models = len(models)
    sample_count = None
    sums = None
    gts = None

    for i, entry in enumerate(models):
        ds = StaticBEVDataset(data_root, split="val", target_size=entry["image_size"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        if sample_count is None:
            sample_count = len(ds)
            sums = torch.zeros((sample_count, 1, 188, 126), dtype=torch.float32)
            gts = torch.zeros((sample_count, 1, 188, 126), dtype=torch.long)
        offset = 0
        for batch in tqdm(loader, desc=f"val [{i+1}/{n_models}: {entry['name']}]"):
            probs = _model_probs(entry, batch, torch.device(next(entry["model"].parameters()).device), tta)
            B = probs.shape[0]
            sums[offset:offset+B] += probs.cpu().float()
            if i == 0:
                gts[offset:offset+B] = batch["gt"]
            offset += B

    avg = sums / n_models
    mask = gts != 255
    return avg[mask], gts[mask]


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
    """Each model predicts test at its own image_size; average then threshold."""
    test_dir = Path(data_root) / SPLIT_DIRS["test"]
    info = pd.read_csv(test_dir / "info.csv", index_col=0)
    n_test = len(info)
    n_models = len(models)
    sums = torch.zeros((n_test, 1, 188, 126), dtype=torch.float32)
    indices = torch.zeros(n_test, dtype=torch.long)

    for i, entry in enumerate(models):
        ds = StaticBEVDataset(data_root, split="test", target_size=entry["image_size"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        offset = 0
        for batch in tqdm(loader, desc=f"test [{i+1}/{n_models}: {entry['name']}]"):
            probs = _model_probs(entry, batch, device, tta)
            B = probs.shape[0]
            sums[offset:offset+B] += probs.cpu().float()
            if i == 0:
                indices[offset:offset+B] = batch["index"]
            offset += B
    avg = sums / n_models

    pred_dir = test_dir / "predicted_static_grids"
    pred_dir.mkdir(exist_ok=True)
    written = 0
    preds = (avg > threshold).numpy().astype(np.int32)
    for k in range(n_test):
        idx = int(indices[k].item())
        rel = info.iloc[idx]["predicted_occupancy_grid"]
        out_path = Path(data_root) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, preds[k])
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
