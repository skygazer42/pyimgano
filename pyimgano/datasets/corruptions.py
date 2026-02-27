from __future__ import annotations

"""Robustness corruptions dataset wrapper.

This dataset is useful when you want to evaluate a detector under deterministic
image corruptions (lighting/jpeg/blur/etc.) in a torch-style pipeline.

It is deliberately simple and builds on `pyimgano.robustness.corruptions`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset


def _read_u8_bgr(path: str | Path) -> np.ndarray:
    import cv2  # local import

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return np.asarray(img, dtype=np.uint8)


def _read_mask_u8(path: str | Path) -> np.ndarray:
    import cv2  # local import

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return (np.asarray(m, dtype=np.uint8) > 0).astype(np.uint8)


@dataclass(frozen=True)
class CorruptionItem:
    image_u8: np.ndarray
    mask_u8: Optional[np.ndarray]
    path: str
    corruption: str
    severity: int
    meta: dict[str, object]


def _apply_corruption(
    image_u8: np.ndarray,
    *,
    mask_u8: Optional[np.ndarray],
    corruption: str,
    severity: int,
    rng: np.random.Generator,
    synthesis_preset: str = "scratch",
    synthesis_blend: str = "alpha",
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, object]]:
    from pyimgano.robustness import corruptions as corr

    name = str(corruption).strip().lower()
    if name in {"lighting", "exposure", "white_balance"}:
        out_img, out_mask = corr.apply_lighting(image_u8, mask=mask_u8, severity=severity, rng=rng)
        return out_img, out_mask, {"name": "lighting"}
    if name in {"jpeg", "jpg"}:
        out_img, out_mask = corr.apply_jpeg(image_u8, mask=mask_u8, severity=severity, rng=rng)
        return out_img, out_mask, {"name": "jpeg"}
    if name in {"blur", "gaussian_blur"}:
        out_img, out_mask = corr.apply_blur(image_u8, mask=mask_u8, severity=severity, rng=rng)
        return out_img, out_mask, {"name": "blur"}
    if name in {"glare"}:
        out_img, out_mask = corr.apply_glare(image_u8, mask=mask_u8, severity=severity, rng=rng)
        return out_img, out_mask, {"name": "glare"}
    if name in {"geo_jitter", "affine"}:
        out_img, out_mask = corr.apply_geo_jitter(image_u8, mask=mask_u8, severity=severity, rng=rng)
        return out_img, out_mask, {"name": "geo_jitter"}
    if name in {"synthesis", "synthesis_preset", "synthetic"}:
        out_img, out_mask = corr.apply_synthesis_preset(
            image_u8,
            mask=mask_u8,
            severity=severity,
            rng=rng,
            preset=str(synthesis_preset),
            blend=str(synthesis_blend),
        )
        return out_img, out_mask, {"name": "synthesis", "preset": str(synthesis_preset)}

    raise ValueError(
        f"Unknown corruption: {corruption!r}. "
        "Supported: lighting, jpeg, blur, glare, geo_jitter, synthesis_preset."
    )


class CorruptionsDataset(Dataset):
    """Wrap images and deterministically apply a single corruption."""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        corruption: str,
        severity: int = 1,
        seed: int = 0,
        mask_paths: Optional[Sequence[str | Path]] = None,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        synthesis_preset: str = "scratch",
        synthesis_blend: str = "alpha",
    ) -> None:
        self.image_paths = [str(p) for p in list(image_paths)]
        self.corruption = str(corruption)
        self.severity = int(severity)
        self.seed = int(seed)
        self.transform = transform
        self.synthesis_preset = str(synthesis_preset)
        self.synthesis_blend = str(synthesis_blend)

        if mask_paths is None:
            self.mask_paths = None
        else:
            mp = [str(p) for p in list(mask_paths)]
            if len(mp) != len(self.image_paths):
                raise ValueError("mask_paths length must match image_paths length")
            self.mask_paths = mp

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> CorruptionItem:
        i = int(idx)
        path = self.image_paths[i]
        img = _read_u8_bgr(path)

        mask = None
        if self.mask_paths is not None:
            mask = _read_mask_u8(self.mask_paths[i])

        rng = np.random.default_rng((self.seed + i) & 0xFFFFFFFF)
        out_img, out_mask, meta = _apply_corruption(
            img,
            mask_u8=mask,
            corruption=self.corruption,
            severity=int(self.severity),
            rng=rng,
            synthesis_preset=str(self.synthesis_preset),
            synthesis_blend=str(self.synthesis_blend),
        )

        if self.transform is not None:
            out_img = self.transform(out_img)

        return CorruptionItem(
            image_u8=np.asarray(out_img, dtype=np.uint8),
            mask_u8=(None if out_mask is None else np.asarray(out_mask, dtype=np.uint8)),
            path=str(path),
            corruption=str(self.corruption),
            severity=int(self.severity),
            meta=dict(meta),
        )


__all__ = ["CorruptionItem", "CorruptionsDataset"]

