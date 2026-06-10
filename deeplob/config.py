"""Typed configuration loaded from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class DataCfg:
    data_dir: str = "data"
    assets: list[str] = field(default_factory=lambda: ["BTCIRT", "USDTIRT"])
    n_levels: int = 10
    norm_lookback_days: int = 5  # previous-N-days z-score window


@dataclass
class LabelCfg:
    horizon_k: int = 6
    alpha_bps: float = 2.0


@dataclass
class WindowCfg:
    lookback_t: int = 100


@dataclass
class SplitCfg:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass
class ModelCfg:
    n_classes: int = 3


@dataclass
class TrainCfg:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    early_stop_patience: int = 10
    num_workers: int = 0
    seed: int = 42


@dataclass
class PathsCfg:
    checkpoint_dir: str = "checkpoints"
    metrics_dir: str = "metrics"
    log_dir: str = "logs"


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    label: LabelCfg = field(default_factory=LabelCfg)
    window: WindowCfg = field(default_factory=WindowCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    @property
    def alpha(self) -> float:
        """Label threshold as a fraction (bps / 1e4)."""
        return self.label.alpha_bps / 1e4


def load_config(path: str) -> Config:
    """Build a :class:`Config` from a YAML file (missing keys fall back to defaults)."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(
        data=DataCfg(**raw.get("data", {})),
        label=LabelCfg(**raw.get("label", {})),
        window=WindowCfg(**raw.get("window", {})),
        split=SplitCfg(**raw.get("split", {})),
        model=ModelCfg(**raw.get("model", {})),
        train=TrainCfg(**raw.get("train", {})),
        paths=PathsCfg(**raw.get("paths", {})),
    )
