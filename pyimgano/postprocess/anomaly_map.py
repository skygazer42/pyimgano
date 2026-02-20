from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class AnomalyMapPostprocess:
    """Post-process an anomaly heatmap for more stable localization."""

    normalize: bool = True
    gaussian_sigma: float = 0.0
    morph_open_ksize: int = 0
    morph_close_ksize: int = 0
    component_threshold: Optional[float] = None
    min_component_area: int = 0

    def __call__(self, anomaly_map: np.ndarray) -> np.ndarray:
        if anomaly_map.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got shape {anomaly_map.shape}")

        processed = anomaly_map.astype(np.float32, copy=True)

        if self.normalize:
            processed = _normalize_minmax(processed)

        if self.gaussian_sigma and self.gaussian_sigma > 0:
            processed = cv2.GaussianBlur(processed, ksize=(0, 0), sigmaX=float(self.gaussian_sigma))

        if self.morph_open_ksize and self.morph_open_ksize > 0:
            processed = _morph(processed, op="open", ksize=self.morph_open_ksize)

        if self.morph_close_ksize and self.morph_close_ksize > 0:
            processed = _morph(processed, op="close", ksize=self.morph_close_ksize)

        if self.component_threshold is not None and self.min_component_area and self.min_component_area > 0:
            processed = _filter_small_components(
                processed,
                threshold=float(self.component_threshold),
                min_area=int(self.min_component_area),
            )

        return processed


def _normalize_minmax(anomaly_map: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    min_val = float(np.min(anomaly_map))
    max_val = float(np.max(anomaly_map))
    denom = max(max_val - min_val, eps)
    return (anomaly_map - min_val) / denom


def _morph(anomaly_map: np.ndarray, *, op: str, ksize: int) -> np.ndarray:
    k = int(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Work in uint8 space for stable morphology, then return float in [0, 1].
    scaled = np.clip(anomaly_map * 255.0, 0.0, 255.0).astype(np.uint8)
    if op == "open":
        out = cv2.morphologyEx(scaled, cv2.MORPH_OPEN, kernel)
    elif op == "close":
        out = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unsupported op: {op}")
    return out.astype(np.float32) / 255.0


def _filter_small_components(anomaly_map: np.ndarray, *, threshold: float, min_area: int) -> np.ndarray:
    binary = (anomaly_map >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return anomaly_map

    keep = np.zeros(num_labels, dtype=bool)
    keep[0] = False  # background
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep[label] = area >= min_area

    mask = keep[labels]
    out = anomaly_map.copy()
    out[~mask] = 0.0
    return out

