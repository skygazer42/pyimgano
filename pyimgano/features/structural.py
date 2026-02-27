from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor
from pyimgano.io.image import read_image


_ErrorMode = Literal["raise", "zeros"]


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
        # For structural signals the RGB/BGR distinction is not critical.
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(f"Unsupported image shape for structural features: {arr.shape}")


def _downscale_max(gray: np.ndarray, *, max_size: int) -> np.ndarray:
    import cv2

    h, w = map(int, gray.shape[:2])
    if max(h, w) <= int(max_size):
        return gray
    scale = float(max_size) / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def _symmetry(gray: np.ndarray) -> tuple[float, float]:
    g = np.asarray(gray, dtype=np.uint8)
    h, w = map(int, g.shape[:2])
    if h < 2 or w < 2:
        return 0.5, 0.5

    # Horizontal symmetry (left/right)
    half_w = w // 2
    left = g[:, :half_w]
    right = g[:, -half_w:]
    right_flipped = right[:, ::-1]
    h_sym = 1.0 - float(np.mean(np.abs(left.astype(np.float32) - right_flipped.astype(np.float32))) / 255.0)

    # Vertical symmetry (top/bottom)
    half_h = h // 2
    top = g[:half_h, :]
    bottom = g[-half_h:, :]
    bottom_flipped = bottom[::-1, :]
    v_sym = 1.0 - float(np.mean(np.abs(top.astype(np.float32) - bottom_flipped.astype(np.float32))) / 255.0)

    return float(np.clip(h_sym, 0.0, 1.0)), float(np.clip(v_sym, 0.0, 1.0))


def _extract_one(gray: np.ndarray) -> np.ndarray:
    import cv2

    g = np.asarray(gray, dtype=np.uint8)
    if g.ndim != 2:
        raise ValueError("Structural feature extraction expects grayscale image")

    # 1) Edge density at two scales
    edges_low = cv2.Canny(g, 50, 100)
    edges_high = cv2.Canny(g, 100, 200)
    edge_density_low = float(np.mean(edges_low > 0))
    edge_density_high = float(np.mean(edges_high > 0))

    # 2) Gradient magnitude statistics
    grad_x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_mean = float(np.mean(grad_mag) / 255.0)
    grad_std = float(np.std(grad_mag) / 255.0)
    grad_p90 = float(np.percentile(grad_mag, 90) / 255.0)

    # 3) Local variance summary (cheap texture proxy)
    ws = 16
    h, w = map(int, g.shape[:2])
    local_vars: list[float] = []
    for y in range(0, max(1, h - ws + 1), ws):
        for x in range(0, max(1, w - ws + 1), ws):
            win = g[y : y + ws, x : x + ws]
            local_vars.append(float(np.var(win.astype(np.float32)) / (255.0 * 255.0)))
    if local_vars:
        lv = np.asarray(local_vars, dtype=np.float64)
        lv_mean = float(np.mean(lv))
        lv_std = float(np.std(lv))
        lv_p90 = float(np.percentile(lv, 90))
    else:
        lv_mean = lv_std = lv_p90 = 0.0

    # 4) Simple line density (horizontal/vertical/diagonal)
    h_kernel = np.ones((1, 20), np.uint8)
    v_kernel = np.ones((20, 1), np.uint8)
    h_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, v_kernel)
    h_density = float(np.mean(h_lines > 0))
    v_density = float(np.mean(v_lines > 0))

    d = 20
    d1_kernel = np.eye(d, dtype=np.uint8)
    d2_kernel = np.fliplr(d1_kernel)
    d1_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, d1_kernel)
    d2_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, d2_kernel)
    d1_density = float(np.mean(d1_lines > 0))
    d2_density = float(np.mean(d2_lines > 0))

    # 5) Symmetry
    h_sym, v_sym = _symmetry(g)

    # 6) Focus / blur proxy
    lap = cv2.Laplacian(g, cv2.CV_32F)
    lap_var = float(np.var(lap) / (255.0 * 255.0))

    feats = np.asarray(
        [
            edge_density_low,
            edge_density_high,
            grad_mean,
            grad_std,
            grad_p90,
            lv_mean,
            lv_std,
            lv_p90,
            h_density,
            v_density,
            d1_density,
            d2_density,
            h_sym,
            v_sym,
            lap_var,
        ],
        dtype=np.float64,
    )
    return feats


@dataclass
@register_feature_extractor(
    "structural",
    tags=("handcrafted", "structural", "edges", "texture", "fast"),
    metadata={
        "description": "Handcrafted structural feature extractor (edges/gradients/symmetry)",
        "output_dim_hint": 15,
    },
)
class StructuralFeaturesExtractor(BaseFeatureExtractor):
    """Structural (shape/edge/texture) features for industrial baselines.

    This is intended as a deterministic and dependency-light alternative to
    heavy learned embeddings, especially for UI/screen change detection and
    simple surface inspection.
    """

    max_size: int = 512
    error_mode: _ErrorMode = "zeros"

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 15), dtype=np.float64)

        rows: list[np.ndarray] = []
        for item in items:
            try:
                if isinstance(item, (str, Path)):
                    img = read_image(str(item), color="bgr")
                elif isinstance(item, np.ndarray):
                    img = item
                else:
                    raise TypeError(
                        "StructuralFeaturesExtractor expects inputs of type str|Path|np.ndarray, "
                        f"got {type(item)}"
                    )

                gray = _to_gray(np.asarray(img))
                gray = _downscale_max(gray, max_size=int(self.max_size))
                feats = _extract_one(gray)
            except Exception:
                if self.error_mode == "raise":
                    raise
                feats = np.zeros((15,), dtype=np.float64)
            rows.append(np.asarray(feats, dtype=np.float64).reshape(-1))

        return np.stack(rows, axis=0)

