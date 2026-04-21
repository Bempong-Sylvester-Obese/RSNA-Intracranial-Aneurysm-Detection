"""Inference utilities for deployment (predictor, Grad-CAM)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GradCAM3D",
    "PredictionResult",
    "Predictor",
    "grad_cam_volume",
    "gradcam_to_rgba_slice",
    "load_checkpoint",
]


def __getattr__(name: str) -> Any:
    if name in ("GradCAM3D", "grad_cam_volume", "gradcam_to_rgba_slice"):
        import rsna_aneurysm.inference.gradcam as _g

        return getattr(_g, name)
    if name in ("PredictionResult", "Predictor", "load_checkpoint"):
        import rsna_aneurysm.inference.predictor as _p

        return getattr(_p, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
