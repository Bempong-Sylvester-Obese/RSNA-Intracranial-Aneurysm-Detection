"""Pydantic response models for the API.

Kept in a single module so the frontend's TypeScript types can stay in sync by
mirroring the shape here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SeriesMetadata(BaseModel):
    """Basic shape / count info about a stored series."""

    series_id: str
    num_files: int = Field(description="Number of DICOM files on disk.")
    num_slices: int = Field(description="Depth of the processed volume (D).")
    height: int
    width: int
    modality: Optional[str] = None


class VesselScore(BaseModel):
    """A single (label, probability) pair plus its class index."""

    key: str
    label: str
    probability: float
    class_index: int


class PredictionResponse(BaseModel):
    presence: float = Field(description="Probability of any aneurysm being present.")
    presence_index: int
    vessels: list[VesselScore]
    gradcam_ready: bool


class UploadResponse(BaseModel):
    series_id: str
    num_files: int


class HealthResponse(BaseModel):
    status: str
    checkpoint_loaded: bool
    device: str
