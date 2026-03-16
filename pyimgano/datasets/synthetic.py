from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from pyimgano.synthesis.synthesizer import AnomalySynthesizer
from pyimgano.utils.optional_deps import require

Dataset = require("torch.utils.data", extra="torch", purpose="torch-backed datasets").Dataset


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
        severity_range: tuple[float, float] = (1.0, 1.0),
        curriculum_progress: float = 1.0,
        roi_mask: Optional[np.ndarray] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.image_paths = [str(p) for p in image_paths]
        self.synthesizer = synthesizer
        self.p_anomaly = float(p_anomaly)
        self.seed = int(seed)
        s0, s1 = (float(severity_range[0]), float(severity_range[1]))
        s0 = float(np.clip(s0, 0.0, 1.0))
        s1 = float(np.clip(s1, 0.0, 1.0))
        if s1 < s0:
            s0, s1 = s1, s0
        self.severity_range = (s0, s1)
        self.curriculum_progress = float(np.clip(float(curriculum_progress), 0.0, 1.0))
        self.roi_mask = roi_mask
        self.transform = transform

    def set_curriculum_progress(self, progress: float) -> None:
        """Update severity curriculum progress in [0,1] (0=easy, 1=hard)."""

        self.curriculum_progress = float(np.clip(float(progress), 0.0, 1.0))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> SyntheticItem:
        path = self.image_paths[int(idx)]
        img = _read_u8_bgr(path)

        rng = np.random.default_rng((self.seed + int(idx)) & 0xFFFFFFFF)
        p = float(np.clip(self.p_anomaly, 0.0, 1.0))

        if p > 0.0 and float(rng.uniform(0.0, 1.0)) < p:
            s0, s1 = self.severity_range
            prog = float(np.clip(self.curriculum_progress, 0.0, 1.0))
            sev = float(s0 + (s1 - s0) * prog * float(rng.uniform(0.0, 1.0)))
            res = self.synthesizer(img, rng=rng, roi_mask=self.roi_mask, severity=sev)
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
