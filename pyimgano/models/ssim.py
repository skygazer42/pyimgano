# -*- coding: utf-8 -*-
"""SSIM template baseline (modernized).

The legacy `ssim_template` implementation focused on UI/screen change detection.
It used ad-hoc training methods and did not follow the unified detector contract.

This module rebuilds `ssim_template` around the native `BaseDetector` contract:
- `fit()` learns 1..K templates from normal images
- `decision_function()` scores by (1 - best SSIM)
- `BaseDetector` handles thresholding + `predict()` semantics

Higher score => more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from pyimgano.io.image import read_image

from .base_detector import BaseDetector
from .registry import register_model


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32, copy=False)
    if float(np.nanmax(arr_f)) <= 1.0:
        arr_f = arr_f * 255.0
    arr_f = np.clip(arr_f, 0.0, 255.0)
    return arr_f.astype(np.uint8)


def _to_gray(img: np.ndarray) -> np.ndarray:
    import cv2

    arr = _ensure_uint8(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _resize_gray(gray: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(size_hw[0]), int(size_hw[1])
    return cv2.resize(np.asarray(gray, dtype=np.uint8), (w, h), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class _TemplateSet:
    templates: list[np.ndarray]
    size_hw: tuple[int, int]


def _select_templates(
    imgs: Sequence[np.ndarray],
    *,
    n_templates: int,
    random_state: Optional[int],
) -> list[np.ndarray]:
    if n_templates <= 1 or len(imgs) <= 1:
        return [np.asarray(imgs[0], dtype=np.uint8)]

    n_templates_eff = min(int(n_templates), len(imgs))

    # Cluster using a small flattened representation.
    small_hw = (64, 64)
    reps = []
    for im in imgs:
        g = _resize_gray(im, size_hw=small_hw)
        reps.append(g.reshape(-1).astype(np.float32) / 255.0)
    X = np.stack(reps, axis=0)

    km = KMeans(n_clusters=n_templates_eff, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    templates: list[np.ndarray] = []
    for k in range(n_templates_eff):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        center = km.cluster_centers_[k].reshape(1, -1)
        d = pairwise_distances(X[idx], center, metric="euclidean").reshape(-1)
        best = int(idx[int(np.argmin(d))])
        templates.append(np.asarray(imgs[best], dtype=np.uint8))

    if not templates:
        templates = [np.asarray(imgs[0], dtype=np.uint8)]
    return templates


@register_model(
    "ssim_template",
    tags=("vision", "classical", "template", "ssim"),
    metadata={
        "description": "SSIM template-match baseline (modernized; native BaseDetector contract)",
        "legacy_name": True,
    },
    overwrite=True,
)
class SSIMTemplateDetector(BaseDetector):
    """Template-match detector based on SSIM.

    Parameters
    ----------
    n_templates:
        Number of templates selected from the normal training set.
    resize_hw:
        Resize all images to a fixed (H,W) before scoring.
    random_state:
        Used only when selecting templates (KMeans).
    """

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        n_templates: int = 1,
        resize_hw: tuple[int, int] = (384, 512),
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_templates = int(n_templates)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.random_state = random_state

        self.templates_: list[np.ndarray] | None = None

    def _load_and_preprocess(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if isinstance(item, (str, Path)):
            gray = read_image(str(item), color="gray")
        elif isinstance(item, np.ndarray):
            gray = _to_gray(item)
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")
        return _resize_gray(gray, size_hw=self.resize_hw)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        items = list(X)
        if not items:
            raise ValueError("Training set cannot be empty")

        self._set_n_classes(y)

        imgs = [self._load_and_preprocess(it) for it in items]
        self.templates_ = _select_templates(
            imgs,
            n_templates=int(self.n_templates),
            random_state=self.random_state,
        )

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        if self.templates_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        from pyimgano.utils.optional_deps import require

        skmetrics = require("skimage.metrics", extra="skimage", purpose="SSIM similarity metric")
        ssim = skmetrics.structural_similarity

        items = list(X)
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, it in enumerate(items):
            img = self._load_and_preprocess(it)
            best = -1.0
            for tmpl in self.templates_:
                sim = float(ssim(img, tmpl, data_range=255))
                if sim > best:
                    best = sim
            scores[i] = 1.0 - best
        return scores.reshape(-1)
