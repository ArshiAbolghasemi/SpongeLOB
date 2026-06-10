"""Shared logger factory: console + rotating-free file handler."""

from __future__ import annotations

import logging
import os
from datetime import datetime

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "deeplob", log_dir: str | None = None) -> logging.Logger:
    """Return a configured logger, writing to console and (optionally) a timestamped file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(_FMT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{stamp}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
