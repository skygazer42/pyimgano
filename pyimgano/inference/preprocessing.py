from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pyimgano.preprocessing.industrial_presets import (
    IlluminationContrastKnobs,
    apply_illumination_contrast,
)


ImageInput = str | Path | np.ndarray


def _load_rgb_u8_hwc_from_path(path: str | Path) -> NDArray[np.uint8]:
    img = Image.open(str(path)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image shape (H,W,3), got {arr.shape} for {path}")
    return np.ascontiguousarray(arr)


def parse_illumination_contrast_knobs(payload: Mapping[str, Any]) -> IlluminationContrastKnobs:
    """Parse an illumination/contrast preprocessing payload into a knobs object.

    This is used by `pyimgano-infer` to turn `infer_config.json` payloads into
    a stable runtime configuration.
    """

    defaults = IlluminationContrastKnobs()

    def _bool(key: str, default: bool) -> bool:
        if key not in payload or payload[key] is None:
            return bool(default)
        v = payload[key]
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(int(v))
        raise ValueError(f"preprocessing.illumination_contrast.{key} must be a boolean, got {v!r}")

    def _float(key: str, default: float) -> float:
        if key not in payload or payload[key] is None:
            return float(default)
        try:
            return float(payload[key])
        except Exception as exc:  # noqa: BLE001 - config boundary
            raise ValueError(
                f"preprocessing.illumination_contrast.{key} must be a float, got {payload[key]!r}"
            ) from exc

    def _optional_float(key: str) -> float | None:
        if key not in payload or payload[key] is None:
            return None
        try:
            return float(payload[key])
        except Exception as exc:  # noqa: BLE001 - config boundary
            raise ValueError(
                f"preprocessing.illumination_contrast.{key} must be a float or null, got {payload[key]!r}"
            ) from exc

    def _int_pair(key: str, default: tuple[int, int]) -> tuple[int, int]:
        if key not in payload or payload[key] is None:
            return (int(default[0]), int(default[1]))
        v = payload[key]
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(
                f"preprocessing.illumination_contrast.{key} must be a list of length 2, got {v!r}"
            )
        try:
            a = int(v[0])
            b = int(v[1])
        except Exception as exc:  # noqa: BLE001 - config boundary
            raise ValueError(
                f"preprocessing.illumination_contrast.{key} must contain ints, got {v!r}"
            ) from exc
        if a <= 0 or b <= 0:
            raise ValueError(
                f"preprocessing.illumination_contrast.{key} must be positive ints, got {(a, b)}"
            )
        return (a, b)

    wb_raw = str(payload.get("white_balance", defaults.white_balance)).strip().lower()
    if wb_raw in ("", "none"):
        wb = "none"
    elif wb_raw in ("gray_world", "gray-world", "grayworld"):
        wb = "gray_world"
    elif wb_raw in ("max_rgb", "max-rgb", "maxrgb"):
        wb = "max_rgb"
    else:
        raise ValueError(
            "preprocessing.illumination_contrast.white_balance must be one of: none|gray_world|max_rgb"
        )

    cutoff = _float("homomorphic_cutoff", defaults.homomorphic_cutoff)
    if not (0.0 < float(cutoff) <= 1.0):
        raise ValueError("preprocessing.illumination_contrast.homomorphic_cutoff must be in (0,1].")

    clahe_clip_limit = _float("clahe_clip_limit", defaults.clahe_clip_limit)
    if float(clahe_clip_limit) <= 0.0:
        raise ValueError("preprocessing.illumination_contrast.clahe_clip_limit must be > 0.")

    gamma = _optional_float("gamma")
    if gamma is not None and float(gamma) <= 0.0:
        raise ValueError("preprocessing.illumination_contrast.gamma must be > 0 or null.")

    lower_p = _float("contrast_lower_percentile", defaults.contrast_lower_percentile)
    upper_p = _float("contrast_upper_percentile", defaults.contrast_upper_percentile)
    if not (0.0 <= float(lower_p) <= 100.0 and 0.0 <= float(upper_p) <= 100.0 and lower_p < upper_p):
        raise ValueError(
            "preprocessing.illumination_contrast contrast percentiles must satisfy 0<=lower<upper<=100."
        )

    return IlluminationContrastKnobs(
        white_balance=str(wb),
        homomorphic=_bool("homomorphic", defaults.homomorphic),
        homomorphic_cutoff=float(cutoff),
        homomorphic_gamma_low=_float("homomorphic_gamma_low", defaults.homomorphic_gamma_low),
        homomorphic_gamma_high=_float("homomorphic_gamma_high", defaults.homomorphic_gamma_high),
        homomorphic_c=_float("homomorphic_c", defaults.homomorphic_c),
        homomorphic_per_channel=_bool(
            "homomorphic_per_channel", defaults.homomorphic_per_channel
        ),
        clahe=_bool("clahe", defaults.clahe),
        clahe_clip_limit=float(clahe_clip_limit),
        clahe_tile_grid_size=_int_pair("clahe_tile_grid_size", defaults.clahe_tile_grid_size),
        gamma=(float(gamma) if gamma is not None else None),
        contrast_stretch=_bool("contrast_stretch", defaults.contrast_stretch),
        contrast_lower_percentile=float(lower_p),
        contrast_upper_percentile=float(upper_p),
    )


class PreprocessingDetector:
    """Wrap a detector to apply image preprocessing before scoring/maps.

    This wrapper is intended for CLI/workbench path-based flows when you want
    to apply preprocessing but the detector consumes numpy images.
    """

    def __init__(
        self,
        *,
        detector: Any,
        illumination_contrast: IlluminationContrastKnobs | None = None,
    ) -> None:
        self.detector = detector
        self.illumination_contrast = illumination_contrast

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - thin delegation
        return getattr(self.detector, name)

    def _preprocess_item(self, item: ImageInput) -> NDArray[np.uint8]:
        if isinstance(item, (str, Path)):
            arr = _load_rgb_u8_hwc_from_path(item)
        else:
            arr = np.asarray(item)
            if arr.dtype != np.uint8:
                raise TypeError(f"Expected uint8 numpy image input, got dtype={arr.dtype}")
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected image shape (H,W,3), got {arr.shape}")
            arr = np.ascontiguousarray(arr)

        knobs = self.illumination_contrast
        if knobs is not None:
            arr = np.asarray(apply_illumination_contrast(arr, knobs=knobs), dtype=np.uint8)
        return arr

    def _preprocess_batch(self, X: Iterable[ImageInput]) -> list[NDArray[np.uint8]]:
        return [self._preprocess_item(item) for item in list(X)]

    def fit(self, X, y=None, **kwargs):  # noqa: ANN001 - sklearn-style boundary
        batch = self._preprocess_batch(X)
        try:
            return self.detector.fit(batch, y=y, **kwargs)
        except TypeError:
            return self.detector.fit(batch)

    def _call_decision_function(self, batch: Sequence[NDArray[np.uint8]]):
        try:
            return self.detector.decision_function(batch)
        except Exception as exc:
            try:
                stacked = np.stack([np.asarray(x) for x in batch], axis=0)
            except Exception:
                raise exc
            return self.detector.decision_function(stacked)

    def decision_function(self, X: Iterable[ImageInput]):  # noqa: ANN001 - sklearn-style boundary
        batch = self._preprocess_batch(X)
        return self._call_decision_function(batch)

    def predict_anomaly_map(self, X: Iterable[ImageInput]):  # noqa: ANN001 - optional protocol
        if not hasattr(self.detector, "predict_anomaly_map"):
            raise AttributeError("Underlying detector has no predict_anomaly_map")
        batch = self._preprocess_batch(X)
        try:
            return self.detector.predict_anomaly_map(batch)
        except Exception as exc:
            try:
                stacked = np.stack([np.asarray(x) for x in batch], axis=0)
            except Exception:
                raise exc
            return self.detector.predict_anomaly_map(stacked)

    def get_anomaly_map(self, item: ImageInput):  # noqa: ANN001 - optional protocol
        if not hasattr(self.detector, "get_anomaly_map"):
            raise AttributeError("Underlying detector has no get_anomaly_map")
        arr = self._preprocess_item(item)
        return self.detector.get_anomaly_map(arr)
