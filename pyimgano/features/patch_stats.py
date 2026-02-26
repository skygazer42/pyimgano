"""Patch-grid statistics feature extractor."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


def _load_image(item: Any) -> np.ndarray:
    from pathlib import Path

    if isinstance(item, (str, Path)):
        from pyimgano.io.image import read_image

        return np.asarray(read_image(item, color="bgr"))
    return np.asarray(item)


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return gray.astype(np.float32) / 255.0


@register_feature_extractor(
    "patch_stats",
    tags=("image", "statistics"),
    metadata={"description": "Patch-grid stats (mean/std/skew/kurt per patch)"},
)
class PatchStatsExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        grid: tuple[int, int] = (4, 4),
        stats: Sequence[str] = ("mean", "std", "skew", "kurt"),
        resize_hw: tuple[int, int] | None = (128, 128),
        eps: float = 1e-12,
    ) -> None:
        self.grid = (int(grid[0]), int(grid[1]))
        self.stats = tuple(str(s).lower() for s in stats)
        self.resize_hw = resize_hw
        self.eps = float(eps)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        import cv2

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        gr, gc = int(self.grid[0]), int(self.grid[1])
        if gr <= 0 or gc <= 0:
            raise ValueError("grid must be positive")

        feats: list[np.ndarray] = []
        for item in items:
            img = _load_image(item)
            if self.resize_hw is not None:
                h, w = int(self.resize_hw[0]), int(self.resize_hw[1])
                img = cv2.resize(np.asarray(img), (w, h), interpolation=cv2.INTER_AREA)

            gray = _to_gray_f32(img)
            h, w = int(gray.shape[0]), int(gray.shape[1])

            ys = np.linspace(0, h, gr + 1, dtype=int)
            xs = np.linspace(0, w, gc + 1, dtype=int)

            vec: list[float] = []
            for r in range(gr):
                for c in range(gc):
                    patch = gray[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
                    x = patch.reshape(-1).astype(np.float64)
                    if x.size == 0:
                        mu = 0.0
                        sd = 0.0
                        skew = 0.0
                        kurt = 0.0
                    else:
                        mu = float(np.mean(x))
                        sd = float(np.std(x))
                        sd_safe = max(sd, float(self.eps))
                        z = (x - mu) / sd_safe
                        skew = float(np.mean(z**3))
                        kurt = float(np.mean(z**4) - 3.0)

                    for s in self.stats:
                        if s == "mean":
                            vec.append(mu)
                        elif s == "std":
                            vec.append(sd)
                        elif s == "skew":
                            vec.append(skew)
                        elif s in {"kurt", "kurtosis"}:
                            vec.append(kurt)
                        else:
                            raise ValueError(f"Unknown stat: {s!r}")

            feats.append(np.asarray(vec, dtype=np.float32))

        return np.stack(feats, axis=0)

