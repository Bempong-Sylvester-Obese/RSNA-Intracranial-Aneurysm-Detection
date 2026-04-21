"""Pick best available PyTorch device: CUDA, MPS (Apple Silicon), or CPU."""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Prefer NVIDIA GPU, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_pin_memory(device: torch.device) -> bool:
    """Host pinned memory helps only when copying to CUDA."""
    return device.type == "cuda"
