"""Runtime settings for the webapp backend.

Values come from environment variables so the app can be launched in dev with
``uvicorn`` without a config file. Defaults keep everything under the repo so
contributors can run the stack end-to-end without extra setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    return Path(value).expanduser() if value else default


def _optional_env_path(key: str) -> Path | None:
    value = os.environ.get(key)
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    """Backend runtime configuration.

    Environment variables:

    * ``RSNA_CHECKPOINT`` – path to a ``.pt`` checkpoint. Optional; when unset
      the predictor still works but returns predictions from a randomly-initialised
      model (useful for UI development).
    * ``RSNA_STORAGE_DIR`` – directory where uploaded series are extracted.
    * ``RSNA_MAX_UPLOAD_MB`` – maximum total upload size in megabytes.
    * ``RSNA_MAX_SLICES_PER_SERIES`` – hard cap on the number of DICOM files
      kept after extraction, matched against the DICOM processor's depth.
    * ``RSNA_CORS_ORIGINS`` – comma-separated list of allowed CORS origins.
    """

    checkpoint_path: Path | None = field(
        default_factory=lambda: _optional_env_path("RSNA_CHECKPOINT")
    )
    storage_dir: Path = field(
        default_factory=lambda: _env_path(
            "RSNA_STORAGE_DIR", _REPO_ROOT / "webapp" / "backend" / "storage"
        )
    )
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("RSNA_MAX_UPLOAD_MB", 512) * 1024 * 1024
    )
    max_slices_per_series: int = field(
        default_factory=lambda: _env_int("RSNA_MAX_SLICES_PER_SERIES", 1024)
    )
    max_file_size_bytes: int = field(
        default_factory=lambda: _env_int("RSNA_MAX_FILE_MB", 64) * 1024 * 1024
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "RSNA_CORS_ORIGINS",
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ],
        )
    )

    def resolve_checkpoint(self) -> Path | None:
        cp = self.checkpoint_path
        if cp is None:
            return None
        cp = Path(cp)
        if str(cp) in {"", "."}:
            return None
        if not cp.is_file():
            return None
        return cp


def get_settings() -> Settings:
    """Factory used by FastAPI dependencies (cheap, dataclass).

    Re-read each call so tests can monkeypatch env vars; the actual predictor
    is cached separately.
    """
    return Settings()
