"""Array-backed datasets for in-memory inference/training.

These datasets are primarily used to support numpy-first industrial workflows
where images are already decoded in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from pyimgano.utils.optional_deps import require

torch = require("torch", extra="torch", purpose="torch-backed datasets")
Dataset = require("torch.utils.data", extra="torch", purpose="torch-backed datasets").Dataset


class VisionArrayDataset(Dataset):
    """Dataset backed by a sequence of canonical numpy images.

    Expected input images:
    - RGB
    - uint8
    - HWC
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        fallback_shape: tuple[int, int, int] = (3, 224, 224),
    ) -> None:
        self.images = list(images)
        self.transform = transform
        self.target_transform = target_transform
        self.fallback_shape = fallback_shape

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got {type(image)}")
            if image.dtype != np.uint8:
                raise ValueError(f"Expected dtype=uint8, got {image.dtype}")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {image.shape}")

            if self.transform is None:
                out = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
                out = out / 255.0
            else:
                pil = Image.fromarray(image, mode="RGB")
                out = self.transform(pil)

            target = out
            if self.target_transform is not None:
                target = self.target_transform(target)
            return out, target
        except Exception as exc:  # noqa: BLE001 - parity with VisionImageDataset
            print(f"加载数组图片时出错 index={idx}: {exc}")
            fallback = torch.zeros(self.fallback_shape, dtype=torch.float32)
            return fallback, fallback.clone()
