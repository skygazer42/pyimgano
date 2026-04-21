# -*- coding: utf-8 -*-
"""Normalized cross-correlation (NCC) pixel-map detector (template matching).

This is an industrial baseline for aligned template inspection:
- fit(): selects 1..K grayscale templates from normal images
- predict_anomaly_map(): local NCC similarity map vs best template
- decision_function(): reduces anomaly map to an image-level score

Scoring convention follows the native `BaseDetector` contract:
**higher score => more anomalous**.
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
    orig_hw: tuple[int, int]
    rep_u8: np.ndarray


def _local_ncc_map(
    query_u8: np.ndarray,
    template_u8: np.ndarray,
    *,
    window_hw: tuple[int, int],
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a local NCC similarity map (higher => more similar)."""

    import cv2

    q = np.asarray(query_u8, dtype=np.float32)
    t = np.asarray(template_u8, dtype=np.float32)
    if q.shape != t.shape:
        raise ValueError(f"query/template shape mismatch: {q.shape} vs {t.shape}")

    kh, kw = int(window_hw[0]), int(window_hw[1])
    if kh < 1 or kw < 1:
        raise ValueError("window_hw must be positive")
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("window_hw must be odd (centered local statistic)")

    ksize = (kw, kh)
    n = float(kh * kw)

    # Box filter gives local sums when applied to raw values.
    border = cv2.BORDER_REFLECT
    sum_q = cv2.boxFilter(q, ddepth=-1, ksize=ksize, borderType=border, normalize=False)
    sum_t = cv2.boxFilter(t, ddepth=-1, ksize=ksize, borderType=border, normalize=False)
    sum_q2 = cv2.boxFilter(q * q, ddepth=-1, ksize=ksize, borderType=border, normalize=False)
    sum_t2 = cv2.boxFilter(t * t, ddepth=-1, ksize=ksize, borderType=border, normalize=False)
    sum_qt = cv2.boxFilter(q * t, ddepth=-1, ksize=ksize, borderType=border, normalize=False)

    num = sum_qt - (sum_q * sum_t) / n
    var_q = sum_q2 - (sum_q * sum_q) / n
    var_t = sum_t2 - (sum_t * sum_t) / n

    var_q = np.maximum(var_q, 0.0)
    var_t = np.maximum(var_t, 0.0)

    denom = np.sqrt(var_q * var_t) + float(eps)
    ncc = num / denom
    ncc = np.clip(ncc, -1.0, 1.0).astype(np.float32)
    return ncc


@register_model(
    "vision_template_ncc_map",
    tags=("vision", "classical", "template", "ncc", "pixel_map"),
    metadata={"description": "Template baseline using local NCC similarity → pixel anomaly map"},
)
class VisionTemplateNCCMapDetector(BaseDetector):
    """Local NCC similarity vs best template, mapped to [0,1] anomaly map."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        n_templates: int = 1,
        resize_hw: tuple[int, int] = (384, 512),
        window_hw: tuple[int, int] = (11, 11),
        random_state: Optional[int] = 42,
        reduction: str = "topk_mean",
        topk: float = 0.01,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_templates = int(n_templates)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.window_hw = (int(window_hw[0]), int(window_hw[1]))
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

    def _prepare(self, item) -> _PreparedImage:  # noqa: ANN001, ANN201
        gray = self._load_gray(item)
        orig_hw = (int(gray.shape[0]), int(gray.shape[1]))
        rep = _resize_gray(gray, size_hw=self.resize_hw)
        return _PreparedImage(orig_hw=orig_hw, rep_u8=np.asarray(rep, dtype=np.uint8))

    # ------------------------------------------------------------------
    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        items = list(cast(Iterable[object], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
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

        prepared = self._prepare(item)
        rep = prepared.rep_u8

        best_sim = -np.inf
        best_ncc: np.ndarray | None = None
        for tmpl in self.templates_:
            ncc = _local_ncc_map(rep, tmpl, window_hw=self.window_hw)
            sim = float(np.mean(ncc))
            if sim > best_sim or best_ncc is None:
                best_sim = sim
                best_ncc = ncc

        if best_ncc is None:
            raise RuntimeError("Internal error: failed to compute a template NCC map.")
        am = (1.0 - best_ncc) * 0.5
        am = np.clip(am, 0.0, 1.0).astype(np.float32)

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


__all__ = ["VisionTemplateNCCMapDetector"]
