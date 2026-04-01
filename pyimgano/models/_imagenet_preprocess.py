from __future__ import annotations

from typing import Any

import numpy as np


def preprocess_imagenet_batch(x: Any):
    """Convert uint8/f32 image batches into ImageNet-normalized torch tensors.

    Accepts either NHWC RGB arrays or NCHW arrays and returns a float32 tensor
    in NCHW layout, normalized by standard ImageNet mean/std.
    """

    import torch

    arr = np.asarray(x)
    if arr.shape[-1] == 3:
        arr = np.transpose(arr, (0, 3, 1, 2))

    arr = arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    arr = (arr - mean) / std

    return torch.from_numpy(arr).float()


__all__ = ["preprocess_imagenet_batch"]
