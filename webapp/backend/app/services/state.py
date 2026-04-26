"""Cached singletons for the FastAPI app (predictor + storage).

Loading the PyTorch model is expensive; we do it once at startup and share it
across requests. A small in-memory cache of ``(series_id, volume, cam)`` is
kept so repeat slice requests don't re-run preprocessing.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from rsna_aneurysm.device import pick_device
from rsna_aneurysm.inference.predictor import PredictionResult, Predictor
from webapp.backend.app.services.storage import SeriesStorage
from webapp.backend.app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SeriesCacheEntry:
    """Cached preprocessing / prediction products for one series."""

    volume: np.ndarray
    result: Optional[PredictionResult] = None
    cams: dict[int, np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cams is None:
            self.cams = {}


class AppState:
    """Holds lazily-initialised singletons."""

    _instance: "AppState | None" = None
    _lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = SeriesStorage(
            storage_dir=settings.storage_dir,
            max_upload_bytes=settings.max_upload_bytes,
            max_file_size_bytes=settings.max_file_size_bytes,
            max_slices_per_series=settings.max_slices_per_series,
        )
        device = pick_device()
        ckpt = settings.resolve_checkpoint()
        self.predictor = Predictor.load(ckpt, device)
        self.checkpoint_path = ckpt
        self._cache: dict[str, SeriesCacheEntry] = {}
        self._cache_lock = threading.Lock()
        logger.info(
            "AppState ready: checkpoint=%s storage=%s device=%s",
            ckpt if ckpt is not None else "<none>",
            self.storage.storage_dir,
            device,
        )

    @classmethod
    def init(cls, settings: Settings) -> "AppState":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            return cls._instance

    @classmethod
    def get(cls) -> "AppState":
        if cls._instance is None:
            raise RuntimeError("AppState has not been initialised.")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached instance (useful for tests)."""
        with cls._lock:
            cls._instance = None

    def get_or_load_volume(self, series_id: str) -> SeriesCacheEntry:
        with self._cache_lock:
            entry = self._cache.get(series_id)
            if entry is not None:
                return entry

        path = self.storage.resolve(series_id)
        volume = self.predictor.load_volume(path)
        entry = SeriesCacheEntry(volume=volume)
        with self._cache_lock:
            self._cache[series_id] = entry
        return entry

    def set_result(self, series_id: str, result: PredictionResult) -> None:
        with self._cache_lock:
            entry = self._cache.get(series_id)
            if entry is None:
                return
            entry.result = result

    def set_cam(self, series_id: str, class_index: int, cam: np.ndarray) -> None:
        with self._cache_lock:
            entry = self._cache.get(series_id)
            if entry is None:
                return
            entry.cams[class_index] = np.asarray(cam, dtype=np.float32)

    def drop(self, series_id: str) -> None:
        with self._cache_lock:
            self._cache.pop(series_id, None)
