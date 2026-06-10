"""End-to-end DeepLOB pipeline: build data -> train -> validate -> test.

Usage:
    uv run python scripts/train_deeplob.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from deeplob.config import load_config
from deeplob.dataset import build_datasets, class_weights
from deeplob.logging_utils import get_logger
from deeplob.model import DeepLOB
from deeplob.test import run_test
from deeplob.train import train_model


def pick_device(logger) -> torch.device:
    """Prefer CUDA GPU, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # autotune convs for fixed input shape
        logger.info("CUDA available -> using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("CUDA not available -> using Apple MPS")
        return torch.device("mps")
    logger.info("CUDA/MPS not available -> using CPU")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepLOB on LOB snapshots.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name", default="deeplob")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("deeplob", cfg.paths.log_dir)
    set_seed(cfg.train.seed)

    device = pick_device(logger)
    logger.info("device=%s | config=%s", device, args.config)
    logger.info(
        "label: k=%d alpha=%.1fbps | window T=%d | assets=%s",
        cfg.label.horizon_k,
        cfg.label.alpha_bps,
        cfg.window.lookback_t,
        cfg.data.assets,
    )

    datasets, n_features = build_datasets(cfg, logger)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
    )

    model = DeepLOB(n_features=n_features, n_classes=cfg.model.n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model: DeepLOB(n_features=%d) | %d parameters", n_features, n_params)

    weights = class_weights(datasets["train"], cfg.model.n_classes)
    logger.info("train class weights: %s", weights.tolist())

    train_model(
        model, train_loader, val_loader, cfg, device, logger, args.name, weights
    )
    run_test(cfg, datasets["test"], n_features, device, logger, args.name)


if __name__ == "__main__":
    main()
