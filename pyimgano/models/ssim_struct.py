# -*- coding: utf-8 -*-
"""Structural SSIM template baseline (modernized).

`ssim_struct` is a UI/screen-change baseline that focuses on structure rather
than color. We implement it as SSIM over a structural representation (edges).

This keeps the registry name stable but aligns the model with the native
`BaseDetector` contract:
- `fit()` selects templates from normal data
- `decision_function()` scores by (1 - best SSIM)
- `predict()` uses contamination-derived thresholding
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def _resize(gray: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(size_hw[0]), int(size_hw[1])
    return cv2.resize(np.asarray(gray, dtype=np.uint8), (w, h), interpolation=cv2.INTER_AREA)


def _edges(gray: np.ndarray, *, t1: int, t2: int) -> np.ndarray:
    import cv2

    e = cv2.Canny(np.asarray(gray, dtype=np.uint8), int(t1), int(t2))
    # Make sure template values are in [0,255] for SSIM.
    return np.asarray(e, dtype=np.uint8)


def _select_templates(
    imgs: list[np.ndarray], *, n_templates: int, random_state: Optional[int]
) -> list[np.ndarray]:
    if n_templates <= 1 or len(imgs) <= 1:
        return [np.asarray(imgs[0], dtype=np.uint8)]

    n_templates_eff = min(int(n_templates), len(imgs))

    small_hw = (64, 64)
    X = np.stack(
        [(_resize(im, size_hw=small_hw).reshape(-1).astype(np.float32) / 255.0) for im in imgs],
        axis=0,
    )

    km = KMeans(n_clusters=n_templates_eff, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    templates: list[np.ndarray] = []
    for k in range(n_templates_eff):
        idx = np.nonzero(labels == k)[0]
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
    "ssim_struct",
    tags=("vision", "classical", "template", "ssim", "structural"),
    metadata={
        "description": "Structural SSIM template-match baseline (edges; modernized)",
        "legacy_name": True,
        "paper": "Image Quality Assessment: From Error Visibility to Structural Similarity",
        "year": 2004,
    },
    overwrite=True,
)
class SSIMStructDetector(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.01,
        n_templates: int = 1,
        resize_hw: tuple[int, int] = (384, 512),
        canny_threshold1: int = 100,
        canny_threshold2: int = 200,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_templates = int(n_templates)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.canny_threshold1 = int(canny_threshold1)
        self.canny_threshold2 = int(canny_threshold2)
        self.random_state = random_state

        self.templates_: list[np.ndarray] | None = None

    def _load_and_preprocess(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if isinstance(item, (str, Path)):
            gray = read_image(str(item), color="gray")
        elif isinstance(item, np.ndarray):
            gray = _to_gray(item)
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")

        gray = _resize(gray, size_hw=self.resize_hw)
        return _edges(gray, t1=self.canny_threshold1, t2=self.canny_threshold2)

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        items = list(x)
        if not items:
            raise ValueError("Training set cannot be empty")

        self._set_n_classes(y)

        imgs = [self._load_and_preprocess(it) for it in items]
        self.templates_ = _select_templates(
            imgs, n_templates=int(self.n_templates), random_state=self.random_state
        )

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        if self.templates_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        from pyimgano.utils.optional_deps import require

        skmetrics = require("skimage.metrics", extra="skimage", purpose="SSIM similarity metric")
        ssim = skmetrics.structural_similarity

        items = list(x)
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
