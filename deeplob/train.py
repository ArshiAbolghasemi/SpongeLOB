"""Training loop with validation, early stopping, checkpointing, and metric logging."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .evaluate import compute_metrics, eval_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
    name: str = "deeplob",
    class_weight: torch.Tensor | None = None,
) -> dict:
    """Train ``model``, checkpoint the best val-loss weights, and return the run history."""
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    weight = class_weight.to(device) if class_weight is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    history: list[dict] = []
    best_val = np.inf
    best_epoch = 0
    since_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = datetime.now()
        model.train()
        batch_losses = []
        pbar = tqdm(
            train_loader, desc=f"{name} ep {epoch:3d}/{cfg.train.epochs}", leave=False
        )
        for x, y in pbar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        train_loss = float(np.mean(batch_losses))

        val_loss, val_preds, val_targets = eval_loss(
            model, val_loader, device, criterion
        )
        val_metrics = compute_metrics(val_preds, val_targets)
        secs = (datetime.now() - t0).total_seconds()

        logger.info(
            "epoch %3d/%d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macroF1=%.4f | %.1fs",
            epoch,
            cfg.train.epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            secs,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_loss < best_val:
            best_val, best_epoch, since_improve = val_loss, epoch, 0
            torch.save(model.state_dict(), best_path)
            logger.info("  ↳ new best (val_loss=%.4f) saved to %s", best_val, best_path)
        else:
            since_improve += 1
            if since_improve >= cfg.train.early_stop_patience:
                logger.info(
                    "early stopping at epoch %d (best epoch %d)", epoch, best_epoch
                )
                break

    os.makedirs(cfg.paths.metrics_dir, exist_ok=True)
    hist_path = os.path.join(cfg.paths.metrics_dir, f"{name}_history.json")
    with open(hist_path, "w") as fh:
        json.dump(
            {"best_epoch": best_epoch, "best_val_loss": best_val, "history": history},
            fh,
            indent=2,
        )
    logger.info(
        "training done: best epoch %d (val_loss=%.4f); history -> %s",
        best_epoch,
        best_val,
        hist_path,
    )
    return {"best_epoch": best_epoch, "best_val_loss": best_val, "history": history}
