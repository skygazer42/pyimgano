from __future__ import annotations

from typing import Literal, Optional

import numpy as np


def to_torch_chw_float(
    image_rgb_u8_hwc: np.ndarray,
    *,
    normalize: Optional[Literal["imagenet"]] = "imagenet",
):
    """Convert canonical ``RGB/u8/HWC`` numpy image to torch ``float32/CHW``.

    Output is in [0, 1] before normalization. If `normalize="imagenet"`, apply
    standard torchvision ImageNet mean/std.
    """

    if not isinstance(image_rgb_u8_hwc, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_rgb_u8_hwc)}")
    if image_rgb_u8_hwc.dtype != np.uint8:
        raise ValueError(f"Expected dtype=uint8, got {image_rgb_u8_hwc.dtype}")
    if image_rgb_u8_hwc.ndim != 3 or image_rgb_u8_hwc.shape[2] != 3:
        raise ValueError(f"Expected shape (H,W,3), got {image_rgb_u8_hwc.shape}")

    import torch

    t = torch.from_numpy(np.ascontiguousarray(image_rgb_u8_hwc)).permute(2, 0, 1).float()
    t = t / 255.0

    if normalize is None:
        return t
    if normalize == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(3, 1, 1)
        return (t - mean) / std

    raise ValueError(f"Unknown normalize mode: {normalize!r}")
