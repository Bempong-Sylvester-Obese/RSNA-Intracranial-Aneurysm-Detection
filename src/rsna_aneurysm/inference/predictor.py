"""Load checkpoints and run volume-level multilabel prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rsna_aneurysm.config import TrainConfig
from rsna_aneurysm.data.dicom import DICOMVolumeProcessor
from rsna_aneurysm.labels import LABEL_COLUMNS, NUM_LABELS, PRESENCE_COL
from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a checkpoint using PyTorch's weights-only loader when available."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support safe weights-only checkpoint loading. "
            "Upgrade PyTorch to load checkpoints securely."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint '{path}': {exc}") from exc

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint '{path}' is missing the required 'model_state_dict' entry."
        )

    return checkpoint


@dataclass
class PredictionResult:
    """Per-label probabilities aligned with ``LABEL_COLUMNS``."""

    labels: dict[str, float]
    presence: float

    @classmethod
    def from_probs_tensor(cls, probs: torch.Tensor) -> PredictionResult:
        p = probs.detach().float().cpu().numpy().reshape(-1)
        if p.shape[0] != NUM_LABELS:
            raise ValueError(f"Expected {NUM_LABELS} probabilities, got {p.shape[0]}")
        labels = {name: float(p[i]) for i, name in enumerate(LABEL_COLUMNS)}
        presence = labels[PRESENCE_COL]
        return cls(labels=labels, presence=presence)


class Predictor:
    """``EfficientAneurysmNet`` wrapper with DICOM volume preprocessing."""

    def __init__(
        self,
        model: EfficientAneurysmNet,
        device: torch.device,
        *,
        target_size: tuple[int, int, int] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.target_size = target_size or TrainConfig().target_size
        self.processor = DICOMVolumeProcessor(target_size=self.target_size)

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str | None,
        device: torch.device,
    ) -> Predictor:
        model = EfficientAneurysmNet(num_classes=NUM_LABELS).to(device)
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            ckpt = load_checkpoint(path, device)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return cls(model, device)

    def load_volume(self, series_dir: Path | str) -> np.ndarray:
        """Load and preprocess a DICOM series directory into ``(D, H, W)`` float32."""
        path = Path(series_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Series directory not found: {path}")
        return self.processor.load_dicom_series(str(path.resolve()))

    def predict_volume(self, volume: np.ndarray) -> PredictionResult:
        """Run inference on a preprocessed volume ``(D, H, W)`` float32 in ``[0, 1]``."""
        if volume.shape != self.target_size:
            raise ValueError(
                f"Volume shape {volume.shape} does not match target {self.target_size}"
            )
        x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0]
        return PredictionResult.from_probs_tensor(probs)

    def predict(self, series_dir: Path | str) -> PredictionResult:
        """Load DICOM series from a directory and return probabilities."""
        path = Path(series_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Series directory not found: {path}")
        volume = self.processor.load_dicom_series(str(path.resolve()))
        return self.predict_volume(volume)
