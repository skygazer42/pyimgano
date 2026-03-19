# -*- coding: utf-8 -*-
"""Per-pixel statistics baselines (template-style) with anomaly maps.

These baselines are designed for industrial inspection where images are roughly
aligned (template-style) and you want a fast, dependency-light pixel-map model.

Models
------
1) ``vision_pixel_mean_absdiff_map``:
   - fit: per-pixel mean template on normal images
   - map: ``mean(|x - mean|) / 255``

2) ``vision_pixel_gaussian_map``:
   - fit: per-pixel mean + std (streaming/Welford)
   - map: per-pixel z-score ``|x - mean| / std``

3) ``vision_pixel_mad_map``:
   - fit: per-pixel median + MAD (robust; stacks a bounded number of images)
   - map: robust z-score ``0.6745 * |x - median| / MAD``

All models follow the native ``BaseDetector`` convention:
**higher score => more anomalous**.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, cast

import numpy as np

from pyimgano.io.image import read_image

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .registry import register_model

CHANNEL_REDUCE_ERROR = "channel_reduce must be one of: max, mean, l2"
EMPTY_TRAINING_SET_ERROR = "Training set cannot be empty"
GET_ANOMALY_MAP_BEFORE_FIT_ERROR = "Detector must be fitted before calling get_anomaly_map"


_Color = Literal["gray", "rgb"]
_Reduction = Literal["max", "mean", "topk_mean"]
_ChannelReduce = Literal["max", "mean", "l2"]


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32, copy=False)
    if float(np.nanmax(arr_f)) <= 1.0:
        arr_f = arr_f * 255.0
    arr_f = np.clip(arr_f, 0.0, 255.0)
    return arr_f.astype(np.uint8)


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    arr = _ensure_uint8(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        # CHW -> HWC
        return np.transpose(arr, (1, 2, 0)).astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape for RGB: {arr.shape}")


def _to_gray_u8(img: np.ndarray, *, assume_numpy_rgb: bool) -> np.ndarray:
    import cv2

    arr = _ensure_uint8(img)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.asarray(arr[:, :, 0], dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return np.asarray(arr[0], dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        code = cv2.COLOR_RGB2GRAY if bool(assume_numpy_rgb) else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(np.asarray(arr, dtype=np.uint8), code)
    if arr.ndim == 3 and arr.shape[0] == 3:
        hwc = np.transpose(arr, (1, 2, 0)).astype(np.uint8, copy=False)
        code = cv2.COLOR_RGB2GRAY if bool(assume_numpy_rgb) else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(hwc, code)
    raise ValueError(f"Unsupported image shape for gray: {arr.shape}")


def _resize_u8(img_u8: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(size_hw[0]), int(size_hw[1])
    return cv2.resize(np.asarray(img_u8, dtype=np.uint8), (w, h), interpolation=cv2.INTER_AREA)


def _resize_map(m: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(size_hw[0]), int(size_hw[1])
    out = cv2.resize(np.asarray(m, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(out, dtype=np.float32)


def _topk_mean(values: np.ndarray, *, topk: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    frac = float(topk)
    if not (0.0 < frac <= 1.0):
        raise ValueError("topk must be in (0,1].")
    k = max(1, int(np.ceil(frac * float(arr.size))))
    k = min(k, int(arr.size))
    top_vals = np.partition(arr, -k)[-k:]
    return float(np.mean(top_vals))


def _reduce_map(m: np.ndarray, *, reduction: _Reduction, topk: float) -> float:
    red = str(reduction).strip().lower()
    if red == "max":
        return float(np.max(m))
    if red == "mean":
        return float(np.mean(m))
    if red == "topk_mean":
        return float(_topk_mean(m, topk=float(topk)))
    raise ValueError("reduction must be one of: max, mean, topk_mean")


def _reduce_channels(z: np.ndarray, *, mode: _ChannelReduce) -> np.ndarray:
    """Reduce a (H,W,C) z-score tensor to a (H,W) anomaly map."""

    m = str(mode).strip().lower()
    if m == "max":
        return np.max(z, axis=2)
    if m == "mean":
        return np.mean(z, axis=2)
    if m == "l2":
        return np.sqrt(np.mean(z * z, axis=2))
    raise ValueError(CHANNEL_REDUCE_ERROR)


@dataclass(frozen=True)
class _Prepared:
    orig_hw: tuple[int, int]
    rep_u8: np.ndarray


class _BasePixelStatsMapDetector(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float,
        resize_hw: tuple[int, int],
        color: _Color,
        reduction: _Reduction,
        topk: float,
        assume_numpy_rgb: bool,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.color = str(color).strip().lower()  # type: ignore[assignment]
        if self.color not in ("gray", "rgb"):
            raise ValueError("color must be one of: gray, rgb")

        self.reduction = str(reduction).strip().lower()  # type: ignore[assignment]
        if self.reduction not in ("max", "mean", "topk_mean"):
            raise ValueError("reduction must be one of: max, mean, topk_mean")

        self.topk = float(topk)
        self.assume_numpy_rgb = bool(assume_numpy_rgb)

    # ------------------------------------------------------------------
    def _load_image(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if isinstance(item, (str, Path)):
            if self.color == "gray":
                return read_image(str(item), color="gray")
            return read_image(str(item), color="rgb")
        if isinstance(item, np.ndarray):
            if self.color == "gray":
                return _to_gray_u8(item, assume_numpy_rgb=bool(self.assume_numpy_rgb))
            return _to_rgb_u8(item)
        raise TypeError(f"Unsupported input type: {type(item)}")

    def _prepare(self, item) -> _Prepared:  # noqa: ANN001, ANN201
        img = self._load_image(item)
        if self.color == "gray":
            img_u8 = np.asarray(img, dtype=np.uint8)
        else:
            img_u8 = _ensure_uint8(img)

        orig_hw = (int(img_u8.shape[0]), int(img_u8.shape[1]))
        rep = _resize_u8(img_u8, size_hw=self.resize_hw)
        return _Prepared(orig_hw=orig_hw, rep_u8=np.asarray(rep, dtype=np.uint8))

    # ------------------------------------------------------------------
    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> np.ndarray:  # noqa: ANN001, ANN201
        items = list(cast(Iterable, resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")))
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(it) for it in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def decision_function(self, x: object = MISSING, **kwargs: object):  # noqa: ANN001, ANN201
        items = list(cast(Iterable, resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")))
        if not items:
            return np.zeros((0,), dtype=np.float64)
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, it in enumerate(items):
            amap = self.get_anomaly_map(it)
            scores[i] = float(_reduce_map(amap, reduction=self.reduction, topk=float(self.topk)))
        return scores.reshape(-1)


@register_model(
    "vision_pixel_mean_absdiff_map",
    tags=("vision", "classical", "template", "pixel_stats", "numpy", "pixel_map"),
    metadata={
        "description": "Per-pixel mean template abs-diff anomaly map (fast aligned baseline)",
        "type": "pixel-first",
    },
)
class VisionPixelMeanAbsDiffMapDetector(_BasePixelStatsMapDetector):
    """Per-pixel mean template abs-diff baseline (map in [0,1])."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        resize_hw: tuple[int, int] = (256, 256),
        color: _Color = "gray",
        reduction: _Reduction = "topk_mean",
        topk: float = 0.01,
        assume_numpy_rgb: bool = True,
    ) -> None:
        super().__init__(
            contamination=float(contamination),
            resize_hw=tuple(resize_hw),
            color=color,
            reduction=reduction,
            topk=float(topk),
            assume_numpy_rgb=bool(assume_numpy_rgb),
        )
        self.mean_template_: np.ndarray | None = None

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201 - sklearn-like signature
        items = list(cast(Iterable, resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError(EMPTY_TRAINING_SET_ERROR)

        self._set_n_classes(y)

        prepared = [self._prepare(it).rep_u8 for it in items]
        stack = np.stack([np.asarray(p, dtype=np.float32) for p in prepared], axis=0)
        self.mean_template_ = np.mean(stack, axis=0).astype(np.float32, copy=False)

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self._process_decision_scores()
        return self

    def get_anomaly_map(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if self.mean_template_ is None:
            raise RuntimeError(GET_ANOMALY_MAP_BEFORE_FIT_ERROR)

        prepared = self._prepare(item)
        rep = prepared.rep_u8.astype(np.float32, copy=False)

        tmpl = np.asarray(self.mean_template_, dtype=np.float32)
        if rep.shape != tmpl.shape:
            raise ValueError(f"Shape mismatch: rep={rep.shape} tmpl={tmpl.shape}")

        if self.color == "gray":
            diff = np.abs(rep - tmpl)
            amap = diff / 255.0
        else:
            diff = np.abs(rep - tmpl)
            amap = np.mean(diff, axis=2) / 255.0

        amap = np.clip(amap, 0.0, 1.0).astype(np.float32, copy=False)

        if prepared.orig_hw != self.resize_hw:
            amap = _resize_map(amap, size_hw=prepared.orig_hw)
            amap = np.clip(amap, 0.0, 1.0).astype(np.float32, copy=False)
        return amap


@register_model(
    "vision_pixel_gaussian_map",
    tags=("vision", "classical", "template", "pixel_stats", "gaussian", "numpy", "pixel_map"),
    metadata={
        "description": "Per-pixel Gaussian baseline (mean+std) anomaly map via z-score",
        "type": "pixel-first",
    },
)
class VisionPixelGaussianMapDetector(_BasePixelStatsMapDetector):
    """Per-pixel Gaussian z-score anomaly map (fast aligned baseline)."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        resize_hw: tuple[int, int] = (256, 256),
        color: _Color = "gray",
        channel_reduce: _ChannelReduce = "max",
        reduction: _Reduction = "topk_mean",
        topk: float = 0.01,
        std_floor: float = 1.0,
        eps: float = 1e-6,
        assume_numpy_rgb: bool = True,
    ) -> None:
        super().__init__(
            contamination=float(contamination),
            resize_hw=tuple(resize_hw),
            color=color,
            reduction=reduction,
            topk=float(topk),
            assume_numpy_rgb=bool(assume_numpy_rgb),
        )
        self.channel_reduce = str(channel_reduce).strip().lower()  # type: ignore[assignment]
        if self.channel_reduce not in ("max", "mean", "l2"):
            raise ValueError(CHANNEL_REDUCE_ERROR)

        self.std_floor = float(std_floor)
        self.eps = float(eps)

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        items = list(cast(Iterable, resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError(EMPTY_TRAINING_SET_ERROR)

        self._set_n_classes(y)

        first = self._prepare(items[0]).rep_u8.astype(np.float64, copy=False)
        mean = np.asarray(first, dtype=np.float64)
        m2 = np.zeros_like(mean, dtype=np.float64)
        count = 1

        for it in items[1:]:
            rep = self._prepare(it).rep_u8.astype(np.float64, copy=False)
            if rep.shape != mean.shape:
                raise ValueError(
                    "Training image shape mismatch after resize. "
                    f"Expected {mean.shape}, got {rep.shape}."
                )

            count += 1
            delta = rep - mean
            mean = mean + delta / float(count)
            delta2 = rep - mean
            m2 = m2 + delta * delta2

        denom = float(max(count - 1, 1))
        var = m2 / denom
        std = np.sqrt(np.maximum(var, 0.0))

        std_floor = max(float(self.std_floor), float(self.eps))
        std = np.maximum(std, std_floor)

        self.mean_ = mean.astype(np.float32, copy=False)
        self.std_ = std.astype(np.float32, copy=False)

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self._process_decision_scores()
        return self

    def get_anomaly_map(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(GET_ANOMALY_MAP_BEFORE_FIT_ERROR)

        prepared = self._prepare(item)
        rep = prepared.rep_u8.astype(np.float32, copy=False)

        mean = np.asarray(self.mean_, dtype=np.float32)
        std = np.asarray(self.std_, dtype=np.float32)
        if rep.shape != mean.shape:
            raise ValueError(f"Shape mismatch: rep={rep.shape} mean={mean.shape}")

        z = np.abs(rep - mean) / (std + float(self.eps))
        if z.ndim == 2:
            amap = z
        elif z.ndim == 3 and z.shape[2] == 3:
            amap = _reduce_channels(z, mode=self.channel_reduce)
        else:
            raise ValueError(f"Unexpected z-score shape: {z.shape}")

        amap = np.asarray(amap, dtype=np.float32)
        if prepared.orig_hw != self.resize_hw:
            amap = _resize_map(amap, size_hw=prepared.orig_hw)
        return np.asarray(amap, dtype=np.float32)


@register_model(
    "vision_pixel_mad_map",
    tags=("vision", "classical", "template", "pixel_stats", "robust", "numpy", "pixel_map"),
    metadata={
        "description": "Per-pixel robust MAD baseline anomaly map (median + MAD z-score)",
        "type": "pixel-first",
    },
)
class VisionPixelMADMapDetector(_BasePixelStatsMapDetector):
    """Per-pixel robust MAD z-score anomaly map (aligned baseline)."""

    def __init__(
        self,
        *,
        contamination: float = 0.01,
        resize_hw: tuple[int, int] = (256, 256),
        color: _Color = "gray",
        channel_reduce: _ChannelReduce = "max",
        reduction: _Reduction = "topk_mean",
        topk: float = 0.01,
        mad_floor: float = 1.0,
        eps: float = 1e-6,
        max_train_images: int = 128,
        random_state: Optional[int] = 0,
        assume_numpy_rgb: bool = True,
        consistency_correction: bool = True,
    ) -> None:
        super().__init__(
            contamination=float(contamination),
            resize_hw=tuple(resize_hw),
            color=color,
            reduction=reduction,
            topk=float(topk),
            assume_numpy_rgb=bool(assume_numpy_rgb),
        )
        self.channel_reduce = str(channel_reduce).strip().lower()  # type: ignore[assignment]
        if self.channel_reduce not in ("max", "mean", "l2"):
            raise ValueError(CHANNEL_REDUCE_ERROR)

        self.mad_floor = float(mad_floor)
        self.eps = float(eps)
        self.max_train_images = int(max_train_images)
        self.random_state = random_state
        self.consistency_correction = bool(consistency_correction)

        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def _maybe_subsample(self, items: Sequence) -> list:  # noqa: ANN001, ANN201
        max_n = int(self.max_train_images)
        if max_n <= 0 or len(items) <= max_n:
            return list(items)
        rng = np.random.default_rng(int(self.random_state) if self.random_state is not None else 0)
        idx = rng.choice(len(items), size=max_n, replace=False)
        return [items[int(i)] for i in np.sort(idx)]

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        items_all = list(cast(Iterable, resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items_all:
            raise ValueError(EMPTY_TRAINING_SET_ERROR)

        self._set_n_classes(y)

        items = self._maybe_subsample(items_all)
        prepared = [self._prepare(it).rep_u8 for it in items]
        stack = np.stack([np.asarray(p, dtype=np.float32) for p in prepared], axis=0)

        med = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - med), axis=0)

        mad_floor = max(float(self.mad_floor), float(self.eps))
        mad = np.maximum(mad, mad_floor)

        self.median_ = np.asarray(med, dtype=np.float32)
        self.mad_ = np.asarray(mad, dtype=np.float32)

        self.decision_scores_ = np.asarray(self.decision_function(items_all), dtype=np.float64)
        self._process_decision_scores()
        return self

    def get_anomaly_map(self, item) -> np.ndarray:  # noqa: ANN001, ANN201
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError(GET_ANOMALY_MAP_BEFORE_FIT_ERROR)

        prepared = self._prepare(item)
        rep = prepared.rep_u8.astype(np.float32, copy=False)

        med = np.asarray(self.median_, dtype=np.float32)
        mad = np.asarray(self.mad_, dtype=np.float32)
        if rep.shape != med.shape:
            raise ValueError(f"Shape mismatch: rep={rep.shape} median={med.shape}")

        z = np.abs(rep - med) / (mad + float(self.eps))
        if self.consistency_correction:
            z = 0.6745 * z

        if z.ndim == 2:
            amap = z
        elif z.ndim == 3 and z.shape[2] == 3:
            amap = _reduce_channels(z, mode=self.channel_reduce)
        else:
            raise ValueError(f"Unexpected z-score shape: {z.shape}")

        amap = np.asarray(amap, dtype=np.float32)
        if prepared.orig_hw != self.resize_hw:
            amap = _resize_map(amap, size_hw=prepared.orig_hw)
        return np.asarray(amap, dtype=np.float32)


__all__ = [
    "VisionPixelMeanAbsDiffMapDetector",
    "VisionPixelGaussianMapDetector",
    "VisionPixelMADMapDetector",
]
