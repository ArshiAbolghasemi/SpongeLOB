"""Load the best checkpoint and report metrics on the held-out test split."""

from __future__ import annotations

import json
import logging
import os

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from .config import Config
from .dataset import WindowDataset
from .evaluate import CLASS_NAMES, compute_metrics, predict
from .model import DeepLOB


def run_test(
    cfg: Config,
    test_ds: WindowDataset,
    n_features: int,
    device: torch.device,
    logger: logging.Logger,
    name: str = "deeplob",
) -> dict:
    """Evaluate the saved best model on ``test_ds`` and persist a metrics report."""
    best_path = os.path.join(cfg.paths.checkpoint_dir, name, "best_model.pt")
    model = DeepLOB(n_features=n_features, n_classes=cfg.model.n_classes).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    logger.info("loaded best model from %s", best_path)

    loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )
    preds, targets, _ = predict(model, loader, device)
    metrics = compute_metrics(preds, targets)

    report = classification_report(
        targets, preds, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0
    )
    logger.info(
        "TEST accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"]
    )
    logger.info("test classification report:\n%s", report)

    os.makedirs(cfg.paths.metrics_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.metrics_dir, f"{name}_test_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("test metrics -> %s", out_path)
    return metrics
