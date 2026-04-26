"""Grad-CAM for ``EfficientAneurysmNet`` (last 128-channel Conv3d block)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet

# Last Conv3d before AdaptiveAvgPool3d in ``features`` (128 -> 128).
_LAST_CONV_LAYER_INDEX = 14


def grad_cam_volume(
    model: EfficientAneurysmNet,
    volume_bc_dhw: torch.Tensor,
    class_idx: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute a 3D class activation map upsampled to the input volume shape.

    ``volume_bc_dhw`` is ``(B, 1, D, H, W)``. Returns ``(D, H, W)`` on CPU float32
    in ``[0, 1]`` after min-max normalization (finite values only).
    """
    if volume_bc_dhw.dim() != 5 or volume_bc_dhw.size(1) != 1:
        raise ValueError("volume_bc_dhw must be (B, 1, D, H, W)")

    layer = model.features[_LAST_CONV_LAYER_INDEX]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _fwd_hook(_module, _inp, out):
        activations.append(out)

    def _bwd_hook(_module, _grad_in, grad_out):
        if grad_out[0] is not None:
            gradients.append(grad_out[0])

    h_fwd = layer.register_forward_hook(_fwd_hook)
    h_bwd = layer.register_full_backward_hook(_bwd_hook)

    model.eval()
    x = volume_bc_dhw.detach().to(device).float().requires_grad_(True)

    try:
        logits = model(x)
        if class_idx < 0 or class_idx >= logits.size(1):
            raise ValueError(f"class_idx {class_idx} out of range for {logits.size(1)} classes")
        score = logits[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=False)
        cam = F.relu(cam)
        cam = cam[0]

        target_spatial = x.shape[2:]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=target_spatial,
            mode="trilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)

        cam = cam.detach().float().cpu()
        finite = torch.isfinite(cam)
        if not finite.any():
            return torch.zeros_like(cam)
        cmin = cam[finite].min()
        cmax = cam[finite].max()
        if (cmax - cmin) < 1e-8:
            return torch.zeros_like(cam)
        cam = (cam - cmin) / (cmax - cmin)
        cam = torch.where(finite, cam, torch.zeros_like(cam))
        return cam
    finally:
        h_fwd.remove()
        h_bwd.remove()


def gradcam_to_rgba_slice(cam_slice: np.ndarray, *, alpha_scale: float = 0.5) -> np.ndarray:
    """Map a 2D CAM ``[0, 1]`` to RGBA uint8 (hot colormap) for alpha compositing."""
    c = np.clip(cam_slice.astype(np.float64), 0.0, 1.0)
    r = c
    g = np.clip((c - 0.25) * 2.0, 0.0, 1.0)
    b = np.clip((c - 0.5) * 2.0, 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    a = (c * 255.0 * float(alpha_scale)).astype(np.uint8)
    return np.concatenate([rgb, a[..., None]], axis=-1)


class GradCAM3D:
    """Context manager that runs Grad-CAM on ``EfficientAneurysmNet`` for one class."""

    def __init__(self, model: EfficientAneurysmNet) -> None:
        self.model = model

    def __enter__(self) -> GradCAM3D:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def compute(
        self,
        volume_dhw: np.ndarray,
        *,
        class_index: int,
        device: torch.device,
    ) -> np.ndarray:
        """Return CAM volume ``(D, H, W)`` float32 in ``[0, 1]``."""
        x = torch.from_numpy(volume_dhw).float().unsqueeze(0).unsqueeze(0)
        cam = grad_cam_volume(self.model, x, class_index, device=device)
        return cam.numpy().astype(np.float32)
