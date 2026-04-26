"""Grad-CAM produces a finite map matching the input volume shape."""

from __future__ import annotations

import numpy as np
import torch

from rsna_aneurysm.inference.gradcam import GradCAM3D, grad_cam_volume
from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet


def test_grad_cam_volume_shape_and_finite() -> None:
    device = torch.device("cpu")
    model = EfficientAneurysmNet(num_classes=14)
    model.eval()
    x = torch.randn(1, 1, 16, 128, 128)
    cam = grad_cam_volume(model, x, class_idx=3, device=device)
    assert cam.shape == (16, 128, 128)
    assert cam.dtype == torch.float32
    assert torch.isfinite(cam).all()


def test_grad_cam_3d_context_matches_numpy() -> None:
    device = torch.device("cpu")
    model = EfficientAneurysmNet(num_classes=14)
    model.eval()
    vol = np.random.randn(16, 128, 128).astype(np.float32)
    with GradCAM3D(model) as cam:
        out = cam.compute(vol, class_index=0, device=device)
    assert out.shape == (16, 128, 128)
    assert np.isfinite(out).all()
