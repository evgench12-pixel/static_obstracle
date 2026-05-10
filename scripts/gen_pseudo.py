"""Run an ensemble on the test set with TTA, save confidence-filtered pseudo labels.

Each pixel of the averaged sigmoid map is binarized:
  prob > high  → 1
  prob < low   → 0
  otherwise    → 255 (ignored by training loss/metric)

The output goes to ``<data_root>/autonomy_yandex_dataset_test/pseudo_labels/``
(file name = same .npy basename as in ``info.csv``).

Usage:
    from scripts.gen_pseudo import main
    main(checkpoints=[
        ("~/static_obstracle/checkpoints/best_lss.pt",     "lss"),
        ("~/static_obstracle/checkpoints/best_lss_r50.pt", "lss_r50"),
    ], low=0.2, high=0.8)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scripts.ensemble import _model_probs, load_models
from src.config import DATA_ROOT, SPLIT_DIRS
from src.dataset import StaticBEVDataset


def main(checkpoints, data_root=str(DATA_ROOT), batch_size=4, num_workers=4,
         tta=True, low=0.2, high=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(checkpoints, device)

    test_dir = Path(data_root) / SPLIT_DIRS["test"]
    info = pd.read_csv(test_dir / "info.csv", index_col=0)
    n_test = len(info)
    n_models = len(models)
    sums = torch.zeros((n_test, 1, 188, 126), dtype=torch.float32)
    indices = torch.zeros(n_test, dtype=torch.long)

    # Each model predicts on test at its native image_size; average resulting BEV probs
    for i, entry in enumerate(models):
        ds = StaticBEVDataset(data_root, split="test", target_size=entry["image_size"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        offset = 0
        for batch in tqdm(loader, desc=f"pseudo [{i+1}/{n_models}: {entry['name']}]"):
            with torch.no_grad():
                probs = _model_probs(entry, batch, device, tta)
            B = probs.shape[0]
            sums[offset:offset+B] += probs.cpu().float()
            if i == 0:
                indices[offset:offset+B] = batch["index"]
            offset += B
    avg = sums / n_models

    out_dir = test_dir / "pseudo_labels"
    out_dir.mkdir(exist_ok=True)
    avg_np = avg.numpy()  # (n_test, 1, 188, 126)
    labels = np.full(avg_np.shape, 255, dtype=np.int32)
    labels[avg_np > high] = 1
    labels[avg_np < low] = 0
    n_zero = int((labels == 0).sum())
    n_one = int((labels == 1).sum())
    n_ignore = int((labels == 255).sum())

    for k in range(n_test):
        idx = int(indices[k].item())
        fname = Path(info.iloc[idx]["predicted_occupancy_grid"]).name
        np.save(out_dir / fname, labels[k])

    total = n_zero + n_one + n_ignore
    print(f"\nDone. Pixel breakdown across {len(info)} samples:")
    print(f"  free (0):    {n_zero/total:.1%}")
    print(f"  occupied (1):{n_one/total:.1%}")
    print(f"  ignored:     {n_ignore/total:.1%}")
    print(f"Pseudo labels at: {out_dir}")
    return out_dir
