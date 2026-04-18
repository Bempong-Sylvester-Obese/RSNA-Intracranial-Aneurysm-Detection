"""Load and preprocess DICOM series into fixed-size volumes."""

from __future__ import annotations

import gc
import logging
import os

import cv2
import numpy as np
import pydicom
from scipy import ndimage

logger = logging.getLogger(__name__)


class DICOMVolumeProcessor:
    """Resize slices to a target 2D size, stack to `max_slices`, optional HU window."""

    def __init__(
        self,
        target_size: tuple[int, int, int] = (16, 128, 128),
        hu_window: tuple[float, float] = (-1000, 1000),
        max_cache_size: int = 50,
    ) -> None:
        self.target_size = target_size
        self.max_slices = target_size[0]
        self.hu_window = hu_window
        self.stats = {"processed": 0, "failed": 0, "dummy": 0}
        self.cache: dict[str, np.ndarray] = {}
        self.max_cache_size = max_cache_size

    def load_dicom_series(self, series_path: str) -> np.ndarray:
        try:
            if series_path in self.cache:
                self.stats["processed"] += 1
                return self.cache[series_path].copy()

            if not os.path.isdir(series_path):
                logger.warning("Series path does not exist: %s", series_path)
                return self._dummy_volume()

            dicom_files = [
                f for f in os.listdir(series_path) if f.lower().endswith(".dcm")
            ][: self.max_slices]
            if not dicom_files:
                logger.warning("No DICOM files in: %s", series_path)
                return self._dummy_volume()

            pixel_arrays: list[np.ndarray] = []
            target_shape = self.target_size[1:]

            for file_name in dicom_files:
                try:
                    file_path = os.path.join(series_path, file_name)
                    ds = pydicom.dcmread(file_path, force=True)
                    if hasattr(ds, "pixel_array"):
                        arr = ds.pixel_array.astype(np.float32)
                        if arr.ndim == 2:
                            arr = self._preprocess_slice(arr, target_shape)
                            pixel_arrays.append(arr)
                        del ds
                    if len(pixel_arrays) >= self.max_slices:
                        break
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load %s: %s", file_name, e)
                    continue

            if not pixel_arrays:
                return self._dummy_volume()

            volume = self._stack_volume(pixel_arrays)

            if len(self.cache) < self.max_cache_size:
                self.cache[series_path] = volume.copy()

            self.stats["processed"] += 1
            return volume

        except Exception as e:  # noqa: BLE001
            logger.error("Error processing %s: %s", series_path, e)
            self.stats["failed"] += 1
            return self._dummy_volume()

    def _preprocess_slice(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if arr.shape != target_shape:
            arr = cv2.resize(
                arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA
            )
        arr = np.clip(arr, *self.hu_window)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        return arr.astype(np.float32)

    def _stack_volume(self, pixel_arrays: list[np.ndarray]) -> np.ndarray:
        while len(pixel_arrays) < self.max_slices:
            pixel_arrays.append(pixel_arrays[-1])
        pixel_arrays = pixel_arrays[: self.max_slices]
        volume = np.stack(pixel_arrays, axis=0).astype(np.float32)
        volume = ndimage.gaussian_filter(volume, sigma=0.3)
        return volume

    def _dummy_volume(self) -> np.ndarray:
        self.stats["dummy"] += 1
        vol = np.random.normal(0.3, 0.1, self.target_size).astype(np.float32)
        return np.clip(vol, 0, 1)

    def clear_cache(self) -> None:
        self.cache.clear()
        gc.collect()
