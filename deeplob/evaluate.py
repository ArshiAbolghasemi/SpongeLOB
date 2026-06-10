"""Inference helpers and classification metrics."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

CLASS_NAMES = ["down", "flat", "up"]


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over ``loader``; return ``(preds, targets, probs)``."""
    model.eval()
    preds, targets, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device, dtype=torch.float32))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        targets.append(y.numpy())
    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(probs, axis=0),
    )


@torch.no_grad()
def eval_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean loss plus ``(preds, targets)`` over ``loader`` (used for validation)."""
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Accuracy, macro-F1, per-class F1, and the confusion matrix."""
    per_class = f1_score(
        targets, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "f1_per_class": {name: float(v) for name, v in zip(CLASS_NAMES, per_class)},
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1, 2]).tolist(),
    }
