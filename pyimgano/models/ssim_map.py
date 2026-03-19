# -*- coding: utf-8 -*-
"""SSIM-based pixel-map detectors (template matching).

Motivation
----------
`ssim_template` / `ssim_struct` are strong industrial baselines for
screen/UI-change and structured template inspection. They are fast and do not
require deep backbones, but historically only produced *image-level* scores.

This module adds pixel-map capable variants:
- `ssim_template_map`: anomaly map is `1 - ssim_map` against best template
- `ssim_struct_map`: same, but SSIM is computed on edge maps (Canny)

Both detectors conform to the native `BaseDetector` contract:
- `fit()` learns 1..K templates from normal data
- `decision_function()` returns 1D scores (higher => more anomalous)
- `predict()` uses contamination-derived thresholding
- `predict_anomaly_map()` returns (N,H,W) float32 maps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from pyimgano.io.image import read_image

from ._legacy_x import MISSING, resolve_legacy_x_keyword
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


def _resize_map(m: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(size_hw[0]), int(size_hw[1])
    out = cv2.resize(np.asarray(m, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(out, dtype=np.float32)


def _topk_mean(values: np.ndarray, *, topk: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    kf = float(topk)
    if not (0.0 < kf <= 1.0):
        raise ValueError("topk must be in (0,1].")
    k = max(1, int(np.ceil(kf * float(arr.size))))
    k = min(k, int(arr.size))
    top_vals = np.partition(arr, -k)[-k:]
    return float(np.mean(top_vals))


def _select_templates(
    imgs: Sequence[np.ndarray],
    *,
    n_templates: int,
    random_state: Optional[int],
) -> list[np.ndarray]:
    if n_templates <= 1 or len(imgs) <= 1:
        return [np.asarray(imgs[0], dtype=np.uint8)]

    n_templates_eff = min(int(n_templates), len(imgs))

    small_hw = (64, 64)
    reps = []
    for im in imgs:
        g = _resize_gray(im, size_hw=small_hw)
        reps.append(g.reshape(-1).astype(np.float32) / 255.0)
    x_repr = np.stack(reps, axis=0)

    km = KMeans(n_clusters=n_templates_eff, random_state=random_state, n_init=10)
    labels = km.fit_predict(x_repr)

    templates: list[np.ndarray] = []
    for k in range(n_templates_eff):
        idx = np.nonzero(labels == k)[0]
        if idx.size == 0:
            continue
        center = km.cluster_centers_[k].reshape(1, -1)
        d = pairwise_distances(x_repr[idx], center, metric="euclidean").reshape(-1)
        best = int(idx[int(np.argmin(d))])
        templates.append(np.asarray(imgs[best], dtype=np.uint8))

    if not templates:
        templates = [np.asarray(imgs[0], dtype=np.uint8)]
    return templates


@dataclass(frozen=True)
class _PreparedImage:
    # Original grayscale size (before resize)
    orig_hw: tuple[int, int]
    # Preprocessed image used for SSIM (resized)
    rep_u8: np.ndarray


class _BaseSSIMMapDetector(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float,
        n_templates: int,
        resize_hw: tuple[int, int],
        random_state: int | None,
        reduction: str,
        topk: float,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_templates = int(n_templates)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.random_state = random_state
        self.reduction = str(reduction)
        self.topk = float(topk)

        self.templates_: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    def _load_gray(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if isinstance(item, (str, Path)):
            return read_image(str(item), color="gray")
        if isinstance(item, np.ndarray):
            return _to_gray(item)
        raise TypeError(f"Unsupported input type: {type(item)}")

    def _preprocess_representation(self, gray_u8: np.ndarray) -> np.ndarray:
        """Override point for derived detectors (e.g., edges)."""

        return np.asarray(gray_u8, dtype=np.uint8)

    def _prepare(self, item) -> _PreparedImage:  # noqa: ANN001, ANN201
        gray = self._load_gray(item)
        orig_hw = (int(gray.shape[0]), int(gray.shape[1]))
        rep = _resize_gray(self._preprocess_representation(gray), size_hw=self.resize_hw)
        return _PreparedImage(orig_hw=orig_hw, rep_u8=np.asarray(rep, dtype=np.uint8))

    # ------------------------------------------------------------------
    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        items = list(
            cast(Iterable[object], resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        )
        if not items:
            raise ValueError("Training set cannot be empty")

        self._set_n_classes(y)

        reps = [self._prepare(it).rep_u8 for it in items]
        self.templates_ = _select_templates(
            reps,
            n_templates=int(self.n_templates),
            random_state=self.random_state,
        )

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self._process_decision_scores()
        return self

    # ------------------------------------------------------------------
    def get_anomaly_map(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if self.templates_ is None:
            raise RuntimeError("Detector must be fitted before calling get_anomaly_map")

        from pyimgano.utils.optional_deps import require

        skmetrics = require("skimage.metrics", extra="skimage", purpose="SSIM anomaly map")
        ssim = skmetrics.structural_similarity

        prepared = self._prepare(item)
        rep = prepared.rep_u8

        best_sim = -1.0
        best_map: np.ndarray | None = None
        for tmpl in self.templates_:
            sim, sim_map = ssim(rep, tmpl, data_range=255, full=True)
            sim_f = float(sim)
            if sim_f > best_sim or best_map is None:
                best_sim = sim_f
                best_map = np.asarray(sim_map, dtype=np.float32)

        assert best_map is not None
        # Convert similarity map to anomaly map (higher => more anomalous).
        am = 1.0 - best_map
        am = np.clip(am, 0.0, 1.0).astype(np.float32)

        # Return in original image size for downstream usability.
        if prepared.orig_hw != self.resize_hw:
            am = _resize_map(am, size_hw=prepared.orig_hw)
            am = np.clip(am, 0.0, 1.0).astype(np.float32)
        return am

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> np.ndarray:
        items = list(
            cast(
                Iterable[object],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(it) for it in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def decision_function(self, x: object = MISSING, **kwargs: object):  # noqa: ANN001, ANN201
        items = list(
            cast(
                Iterable[object],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        if not items:
            return np.zeros((0,), dtype=np.float64)

        red = str(self.reduction).strip().lower()
        if red not in {"max", "mean", "topk_mean"}:
            raise ValueError("reduction must be one of: max, mean, topk_mean")

        scores = np.zeros((len(items),), dtype=np.float64)
        for i, it in enumerate(items):
            amap = self.get_anomaly_map(it)
            if red == "max":
                scores[i] = float(np.max(amap))
            elif red == "mean":
                scores[i] = float(np.mean(amap))
            else:
                scores[i] = float(_topk_mean(amap, topk=float(self.topk)))
        return scores.reshape(-1)


@register_model(
    "ssim_template_map",
    tags=("vision", "classical", "template", "ssim", "pixel_map"),
    metadata={
        "description": "SSIM template detector with pixel anomaly maps (1 - SSIM map)",
        "paper": "Image Quality Assessment: From Error Visibility to Structural Similarity",
        "year": 2004,
    },
)
class SSIMTemplateMapDetector(_BaseSSIMMapDetector):
    """SSIM template matching with pixel-level anomaly maps."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        n_templates: int = 1,
        resize_hw: tuple[int, int] = (384, 512),
        random_state: Optional[int] = 42,
        reduction: str = "topk_mean",
        topk: float = 0.01,
    ) -> None:
        super().__init__(
            contamination=float(contamination),
            n_templates=int(n_templates),
            resize_hw=resize_hw,
            random_state=random_state,
            reduction=str(reduction),
            topk=float(topk),
        )


@register_model(
    "ssim_struct_map",
    tags=("vision", "classical", "template", "ssim", "structural", "pixel_map"),
    metadata={
        "description": "Structural SSIM (edges) with pixel anomaly maps",
        "paper": "Image Quality Assessment: From Error Visibility to Structural Similarity",
        "year": 2004,
    },
)
class SSIMStructMapDetector(_BaseSSIMMapDetector):
    """SSIM over edge maps (Canny) with pixel-level anomaly maps."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        n_templates: int = 1,
        resize_hw: tuple[int, int] = (384, 512),
        canny_threshold1: int = 100,
        canny_threshold2: int = 200,
        random_state: Optional[int] = 42,
        reduction: str = "topk_mean",
        topk: float = 0.01,
    ) -> None:
        super().__init__(
            contamination=float(contamination),
            n_templates=int(n_templates),
            resize_hw=resize_hw,
            random_state=random_state,
            reduction=str(reduction),
            topk=float(topk),
        )
        self.canny_threshold1 = int(canny_threshold1)
        self.canny_threshold2 = int(canny_threshold2)

    def _preprocess_representation(self, gray_u8: np.ndarray) -> np.ndarray:
        import cv2

        e = cv2.Canny(
            np.asarray(gray_u8, dtype=np.uint8),
            int(self.canny_threshold1),
            int(self.canny_threshold2),
        )
        return np.asarray(e, dtype=np.uint8)


__all__ = ["SSIMTemplateMapDetector", "SSIMStructMapDetector"]
