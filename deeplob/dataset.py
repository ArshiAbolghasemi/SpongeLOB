"""Data pipeline: load -> features -> per-asset chronological split -> windowed datasets.

Windows never cross a split boundary, and each window's forward label horizon stays
inside the same split, so the 70/15/15 split is leak-free. Features are normalized with
a rolling **previous-N-days** z-score (DeepLOB convention), which only uses past days and
is therefore also leak-free across the splits.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config
from .features import build_features, rolling_daily_zscore, smoothed_label


class WindowDataset(Dataset):
    """Lazily slices ``(1, T, NF)`` windows out of shared per-asset feature arrays."""

    def __init__(
        self,
        feats: dict[str, np.ndarray],
        labels: dict[str, np.ndarray],
        index: list[tuple[str, int]],
        lookback_t: int,
    ):
        self.feats = feats
        self.labels = labels
        self.index = index
        self.t = lookback_t

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        asset, end = self.index[i]
        window = self.feats[asset][end - self.t + 1 : end + 1]  # (T, NF)
        x = torch.from_numpy(window).unsqueeze(0)  # (1, T, NF)
        y = torch.tensor(int(self.labels[asset][end]))
        return x, y


def _load_asset(data_dir: str, sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ob = pd.read_csv(os.path.join(data_dir, f"{sym}_orderbook.csv"))
    ob["time"] = pd.to_datetime(ob["time"])
    ob = ob.sort_values("time").reset_index(drop=True)

    tr = pd.read_csv(os.path.join(data_dir, f"{sym}_trades.csv"))
    tr["snapshot_time"] = pd.to_datetime(tr["snapshot_time"])
    return ob, tr


def _split_bounds(n: int, cfg: Config) -> dict[str, tuple[int, int]]:
    n_tr = int(n * cfg.split.train)
    n_val = int(n * cfg.split.val)
    return {
        "train": (0, n_tr),
        "val": (n_tr, n_tr + n_val),
        "test": (n_tr + n_val, n),
    }


def _window_index(
    sym: str, bounds: tuple[int, int], labels: np.ndarray, lookback_t: int, k: int
) -> list[tuple[str, int]]:
    lo, hi = bounds
    # window rows [end-T+1, end] inside split; forward label horizon end+k inside split too
    out = []
    for end in range(lo + lookback_t - 1, hi - k):
        if labels[end] != -1:
            out.append((sym, end))
    return out


def build_datasets(
    cfg: Config, logger: logging.Logger | None = None
) -> tuple[dict[str, WindowDataset], int]:
    """Return ``({"train"/"val"/"test": WindowDataset}, n_features)``."""
    log = logger or logging.getLogger("deeplob")
    feats: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    index: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    n_features = 0

    for sym in cfg.data.assets:
        ob, tr = _load_asset(cfg.data.data_dir, sym)
        x, names, mid = build_features(ob, tr, cfg.data.n_levels)
        y = smoothed_label(mid, cfg.label.horizon_k, cfg.alpha)
        n_features = x.shape[1]

        bounds = _split_bounds(len(x), cfg)
        feats[sym] = rolling_daily_zscore(
            x, ob["time"].to_numpy(), cfg.data.norm_lookback_days
        )
        labels[sym] = y

        for split, b in bounds.items():
            idx = _window_index(sym, b, y, cfg.window.lookback_t, cfg.label.horizon_k)
            index[split].extend(idx)
        log.info(
            "%s: %d snapshots, %d features | windows train=%d val=%d test=%d",
            sym,
            len(x),
            n_features,
            *(
                len(
                    _window_index(
                        sym, bounds[s], y, cfg.window.lookback_t, cfg.label.horizon_k
                    )
                )
                for s in ("train", "val", "test")
            ),
        )

    datasets = {
        split: WindowDataset(feats, labels, index[split], cfg.window.lookback_t)
        for split in ("train", "val", "test")
    }
    for split, ds in datasets.items():
        log.info("split %-5s -> %d pooled windows", split, len(ds))
    return datasets, n_features


def class_weights(ds: WindowDataset, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from a dataset's label index (for imbalanced losses)."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for asset, end in ds.index:
        counts[ds.labels[asset][end]] += 1
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
