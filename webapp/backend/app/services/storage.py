"""Filesystem storage for uploaded DICOM series.

A "series" is stored in ``<storage_dir>/<series_id>/`` where ``series_id`` is a
fresh UUID minted per upload — never derived from client input — so there is
no way for a request to address an arbitrary path on disk. Extraction of user
supplied ZIP archives is done with per-entry validation to prevent
`zip-slip <https://snyk.io/research/zip-slip-vulnerability>`_ style attacks.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class StorageError(ValueError):
    """Raised for malformed / unsafe uploads."""


@dataclass(frozen=True)
class StoredSeries:
    """Result of a successful ingest."""

    series_id: str
    path: Path
    num_files: int


def _is_valid_series_id(series_id: str) -> bool:
    try:
        uuid.UUID(series_id)
    except (ValueError, AttributeError):
        return False
    return True


class SeriesStorage:
    """Manage per-series directories under a fixed storage root."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        max_upload_bytes: int,
        max_file_size_bytes: int,
        max_slices_per_series: int,
    ) -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max_upload_bytes
        self.max_file_size_bytes = max_file_size_bytes
        self.max_slices_per_series = max_slices_per_series

    def create_series_dir(self) -> tuple[str, Path]:
        series_id = str(uuid.uuid4())
        path = (self.storage_dir / series_id).resolve()
        path.relative_to(self.storage_dir)
        path.mkdir(parents=True, exist_ok=False)
        return series_id, path

    def resolve(self, series_id: str) -> Path:
        """Return the on-disk path for an existing series, or raise."""
        if not _is_valid_series_id(series_id):
            raise StorageError(f"Invalid series_id: {series_id!r}")
        candidate = (self.storage_dir / series_id).resolve()
        try:
            candidate.relative_to(self.storage_dir)
        except ValueError as exc:
            raise StorageError("Series path escapes storage root") from exc
        if not candidate.is_dir():
            raise FileNotFoundError(f"Unknown series: {series_id}")
        return candidate

    def delete(self, series_id: str) -> None:
        path = self.resolve(series_id)
        shutil.rmtree(path, ignore_errors=True)

    def ingest_zip(self, source: Path) -> StoredSeries:
        """Safely extract ``source`` (a zip file) into a new series directory."""
        series_id, dest = self.create_series_dir()
        try:
            count = _extract_zip_safely(
                source,
                dest,
                max_total_bytes=self.max_upload_bytes,
                max_file_bytes=self.max_file_size_bytes,
                max_files=self.max_slices_per_series,
            )
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise

        if count == 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise StorageError("Archive did not contain any DICOM (.dcm) files.")

        return StoredSeries(series_id=series_id, path=dest, num_files=count)

    def ingest_files(self, files: Iterable[tuple[str, bytes]]) -> StoredSeries:
        """Write the given ``(name, bytes)`` pairs into a new series directory.

        Rejects suspicious names, oversized files, or archives exceeding the
        configured byte budget.
        """
        series_id, dest = self.create_series_dir()
        count = 0
        total = 0
        try:
            for name, data in files:
                safe_name = _sanitise_dicom_name(name)
                if safe_name is None:
                    continue
                if len(data) > self.max_file_size_bytes:
                    raise StorageError(
                        f"DICOM file '{name}' exceeds per-file size limit."
                    )
                total += len(data)
                if total > self.max_upload_bytes:
                    raise StorageError("Upload exceeds total size limit.")
                target = (dest / safe_name).resolve()
                target.relative_to(dest)
                with open(target, "wb") as fh:
                    fh.write(data)
                count += 1
                if count >= self.max_slices_per_series:
                    break
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise

        if count == 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise StorageError("No valid DICOM (.dcm) files were uploaded.")

        return StoredSeries(series_id=series_id, path=dest, num_files=count)


def _sanitise_dicom_name(name: str) -> str | None:
    """Return a safe flat filename for a ``.dcm`` entry, or ``None`` to skip.

    Flattens directory structure (we only care about the files themselves) and
    forbids suspicious components.
    """
    if not name:
        return None
    base = os.path.basename(name.replace("\\", "/"))
    if not base or base in {".", ".."}:
        return None
    if not base.lower().endswith(".dcm"):
        return None
    if any(sep in base for sep in ("/", "\\", "\x00")):
        return None
    return base


def _extract_zip_safely(
    source: Path,
    dest: Path,
    *,
    max_total_bytes: int,
    max_file_bytes: int,
    max_files: int,
) -> int:
    """Extract only ``*.dcm`` entries from ``source`` into ``dest``.

    Rejects absolute paths, path-traversal components, symlinks, oversized
    entries, and archives whose total decompressed size would blow past the
    configured budget.
    """
    count = 0
    total = 0
    dest = dest.resolve()

    with zipfile.ZipFile(source, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            safe_name = _sanitise_dicom_name(info.filename)
            if safe_name is None:
                continue
            if info.file_size > max_file_bytes:
                raise StorageError(
                    f"Zip entry '{info.filename}' exceeds per-file size limit."
                )
            total += info.file_size
            if total > max_total_bytes:
                raise StorageError("Archive exceeds total upload size limit.")

            target = (dest / safe_name).resolve()
            try:
                target.relative_to(dest)
            except ValueError as exc:
                raise StorageError(
                    f"Zip entry '{info.filename}' escapes destination."
                ) from exc

            with zf.open(info, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            count += 1
            if count >= max_files:
                break
    return count
