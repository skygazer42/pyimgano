from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from pyimgano.synthesis.synthesizer import AnomalySynthesizer


def _read_u8_bgr(path: str | Path) -> np.ndarray:
    import cv2  # local import

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return np.asarray(img, dtype=np.uint8)


@dataclass(frozen=True)
class SyntheticItem:
    image_u8: np.ndarray
    label: int
    mask_u8: np.ndarray
    path: str
    meta: dict[str, object]


class SyntheticAnomalyDataset(Dataset):
    """A dataset wrapper that injects synthetic anomalies on-the-fly.

    Notes
    -----
    - Deterministic per-index given the same `seed`.
    - Intended for experimentation and quick industrial pipelines (not a full
      training framework).
    - For multi-modal defect sampling, build the synthesizer with
      `make_preset_mixture([...])` and pass it via `synthesizer=...`.
    """

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        synthesizer: AnomalySynthesizer,
        p_anomaly: float = 0.5,
        seed: int = 0,
        roi_mask: Optional[np.ndarray] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.image_paths = [str(p) for p in list(image_paths)]
        self.synthesizer = synthesizer
        self.p_anomaly = float(p_anomaly)
        self.seed = int(seed)
        self.roi_mask = roi_mask
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> SyntheticItem:
        path = self.image_paths[int(idx)]
        img = _read_u8_bgr(path)

        rng = np.random.default_rng((self.seed + int(idx)) & 0xFFFFFFFF)
        p = float(np.clip(self.p_anomaly, 0.0, 1.0))

        if p > 0.0 and float(rng.uniform(0.0, 1.0)) < p:
            res = self.synthesizer(img, rng=rng, roi_mask=self.roi_mask)
            out_img = res.image_u8
            mask = res.mask_u8
            label = int(res.label)
            meta = dict(res.meta)
        else:
            out_img = np.asarray(img, dtype=np.uint8)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            label = 0
            meta = {"skipped": True}

        if self.transform is not None:
            out_img = self.transform(out_img)

        return SyntheticItem(
            image_u8=np.asarray(out_img, dtype=np.uint8),
            label=int(label),
            mask_u8=np.asarray(mask, dtype=np.uint8),
            path=str(path),
            meta=meta,
        )
