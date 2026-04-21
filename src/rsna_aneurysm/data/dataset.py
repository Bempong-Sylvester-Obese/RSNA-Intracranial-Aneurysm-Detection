"""PyTorch datasets for volume tensors and multi-label targets."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna_aneurysm.data.dicom import DICOMVolumeProcessor
from rsna_aneurysm.labels import ID_COL, LABEL_COLUMNS

logger = logging.getLogger(__name__)


def resolve_series_path(series_dir: str | Path, series_id: str) -> Path:
    """Resolve a series path and ensure it stays under the configured root."""
    root = Path(series_dir).resolve()
    series_id = str(series_id).strip()
    if not series_id:
        raise ValueError("SeriesInstanceUID must be non-empty.")

    if series_id in {".", ".."}:
        raise ValueError(f"Invalid SeriesInstanceUID '{series_id}'.")

    if os.path.isabs(series_id):
        raise ValueError(f"Invalid SeriesInstanceUID '{series_id}': absolute paths are forbidden.")

    separators = {"/", "\\", os.sep}
    if os.altsep:
        separators.add(os.altsep)
    if any(sep in series_id for sep in separators):
        raise ValueError(
            f"Invalid SeriesInstanceUID '{series_id}': path separators are forbidden."
        )

    candidate = (root / series_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Invalid SeriesInstanceUID '{series_id}': path escapes the series directory."
        ) from exc

    return candidate


def filter_dataframe_with_existing_series(
    df: pd.DataFrame, series_dir: str, *, label: str = "train"
) -> pd.DataFrame:
    """Keep rows whose SeriesInstanceUID subfolder exists under series_dir."""
    before = len(df)

    def _exists(uid: object) -> bool:
        try:
            return resolve_series_path(series_dir, str(uid)).is_dir()
        except ValueError:
            logger.warning("strict_paths [%s]: dropping unsafe SeriesInstanceUID '%s'", label, uid)
            return False

    mask = df[ID_COL].apply(_exists)
    out = df.loc[mask].reset_index(drop=True)
    logger.info(
        "strict_paths [%s]: kept %s / %s rows with existing series dirs",
        label,
        len(out),
        before,
    )
    return out


class AneurysmVolumeDataset(Dataset):
    """Multi-label targets aligned with `LABEL_COLUMNS`."""

    def __init__(
        self,
        df: pd.DataFrame,
        series_dir: str,
        processor: DICOMVolumeProcessor,
        target_size: tuple[int, int, int],
        *,
        mode: str = "train",
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.series_dir = series_dir
        self.processor = processor
        self.target_size = target_size
        self.mode = mode

        missing = [c for c in LABEL_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")

        if ID_COL not in self.df.columns:
            raise ValueError(f"Missing id column: {ID_COL}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        series_id = str(row[ID_COL])
        labels = row[list(LABEL_COLUMNS)].astype(np.float32).values.copy()

        series_path = resolve_series_path(self.series_dir, series_id)
        volume = self.processor.load_dicom_series(os.fspath(series_path))

        volume_t = torch.from_numpy(volume).float().unsqueeze(0)
        label_t = torch.from_numpy(labels).float()

        return {
            "volume": volume_t,
            "label": label_t,
            "series_id": series_id,
            "idx": idx,
        }


class SyntheticVolumeDataset(Dataset):
    """Random volumes and labels for tests and smoke training without DICOM."""

    def __init__(
        self,
        n_samples: int,
        target_size: tuple[int, int, int],
        num_labels: int,
        *,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.target_size = target_size
        self.num_labels = num_labels
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        volume = self._rng.random(self.target_size, dtype=np.float32)
        labels = (self._rng.random(self.num_labels) > 0.7).astype(np.float32)
        return {
            "volume": torch.from_numpy(volume).float().unsqueeze(0),
            "label": torch.from_numpy(labels),
            "series_id": f"synthetic_{idx}",
            "idx": idx,
        }
