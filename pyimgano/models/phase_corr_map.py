# -*- coding: utf-8 -*-
"""Phase correlation (translation registration) pixel-map template baseline.

This detector is designed for industrial inspection where small XY misalignment
between the query image and a "golden" template is common.

Algorithm (per image):
1) estimate translation shift between query and each template via phase-corr
2) pick best template (lowest registration error)
3) warp template into query coordinates
4) anomaly map = |query - aligned_template| / 255

Scoring convention follows the native `BaseDetector` contract:
**higher score => more anomalous**.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
    X = np.stack(reps, axis=0)

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


@dataclass(frozen=True)
class _PreparedImage:
    orig_hw: tuple[int, int]
    rep_u8: np.ndarray


def _warp_by_shift(
    img_u8: np.ndarray,
    *,
    shift_yx: tuple[float, float],
) -> np.ndarray:
    import cv2

    dy, dx = float(shift_yx[0]), float(shift_yx[1])
    h, w = img_u8.shape[:2]
    M = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        np.asarray(img_u8, dtype=np.uint8),
        M,
        dsize=(int(w), int(h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.asarray(warped, dtype=np.uint8)


@register_model(
    "vision_phase_correlation_map",
    tags=("vision", "classical", "template", "phase_corr", "pixel_map"),
    metadata={
        "description": "Template baseline using phase correlation alignment + abs-diff anomaly map"
    },
)
class VisionPhaseCorrelationMapDetector(BaseDetector):
    """Phase correlation alignment vs best template, then abs-diff anomaly map."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        n_templates: int = 1,
        resize_hw: tuple[int, int] = (384, 512),
        random_state: Optional[int] = 42,
        reduction: str = "topk_mean",
        topk: float = 0.01,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_templates = int(n_templates)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.random_state = random_state
        self.reduction = str(reduction)
        self.topk = float(topk)
        self.upsample_factor = int(upsample_factor)

        self.templates_: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    def _load_gray(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if isinstance(item, (str, Path)):
            return read_image(str(item), color="gray")
        if isinstance(item, np.ndarray):
            return _to_gray(item)
        raise TypeError(f"Unsupported input type: {type(item)}")

    def _prepare(self, item) -> _PreparedImage:  # noqa: ANN001, ANN201
        gray = self._load_gray(item)
        orig_hw = (int(gray.shape[0]), int(gray.shape[1]))
        rep = _resize_gray(gray, size_hw=self.resize_hw)
        return _PreparedImage(orig_hw=orig_hw, rep_u8=np.asarray(rep, dtype=np.uint8))

    # ------------------------------------------------------------------
    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like signature
        items = list(X)
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
    def _best_template_alignment(
        self, query_rep_u8: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float]]:
        if self.templates_ is None:
            raise RuntimeError("Detector must be fitted before calling _best_template_alignment")

        from pyimgano.utils.optional_deps import require

        skreg = require(
            "skimage.registration", extra="skimage", purpose="phase correlation alignment"
        )
        phase_cross_correlation = skreg.phase_cross_correlation

        query_f = np.asarray(query_rep_u8, dtype=np.float32)

        best_error = np.inf
        best_shift = (0.0, 0.0)
        best_template = self.templates_[0]

        for tmpl in self.templates_:
            tmpl_f = np.asarray(tmpl, dtype=np.float32)
            shift, error, _diffphase = phase_cross_correlation(
                query_f,
                tmpl_f,
                upsample_factor=max(1, int(self.upsample_factor)),
            )
            err_f = float(error)
            if err_f < best_error:
                best_error = err_f
                best_shift = (float(shift[0]), float(shift[1]))  # (dy,dx)
                best_template = tmpl

        warped = _warp_by_shift(best_template, shift_yx=best_shift)
        return warped, best_shift

    def get_anomaly_map(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        prepared = self._prepare(item)
        rep = prepared.rep_u8

        warped_tmpl, _shift = self._best_template_alignment(rep)

        diff = np.abs(np.asarray(rep, dtype=np.float32) - np.asarray(warped_tmpl, dtype=np.float32))
        am = np.clip(diff / 255.0, 0.0, 1.0).astype(np.float32)

        if prepared.orig_hw != self.resize_hw:
            am = _resize_map(am, size_hw=prepared.orig_hw)
            am = np.clip(am, 0.0, 1.0).astype(np.float32)
        return am

    def predict_anomaly_map(self, X: Iterable) -> np.ndarray:  # noqa: ANN001, ANN201
        items = list(X)
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(it) for it in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like signature
        items = list(X)
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


__all__ = ["VisionPhaseCorrelationMapDetector"]
