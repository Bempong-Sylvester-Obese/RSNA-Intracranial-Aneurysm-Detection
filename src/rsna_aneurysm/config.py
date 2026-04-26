"""Training and data configuration (env / defaults)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _default_num_workers() -> int:
    """macOS DataLoader workers often cause hangs; use 0 unless overridden."""
    return 0 if sys.platform == "darwin" else 2


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


@dataclass
class TrainConfig:
    train_csv: Path = field(default_factory=lambda: _env_path("RSNA_TRAIN_CSV", "Data/train.csv"))
    series_dir: Path = field(default_factory=lambda: _env_path("RSNA_SERIES_DIR", "Data/series"))
    localizers_csv: Path | None = None

    target_size: tuple[int, int, int] = (16, 128, 128)
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 0.5
    early_stopping_patience: int = 5
    scheduler_patience: int = 3
    n_folds: int = 5
    current_fold: int = 0
    seed: int = 42
    num_workers: int = field(default_factory=_default_num_workers)
    strict_paths: bool = False
    debug_samples: int | None = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    use_augmentation: bool = True
    synthetic: bool = False
    synthetic_samples: int = 64

    def resolve_paths(self) -> None:
        self.train_csv = Path(self.train_csv)
        self.series_dir = Path(self.series_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.runs_dir = Path(self.runs_dir)
        env_loc = os.environ.get("RSNA_LOCALIZERS_CSV")
        if self.localizers_csv is None and env_loc:
            self.localizers_csv = Path(env_loc)
