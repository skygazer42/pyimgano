from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")


def _make_pattern_image(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[8:20, 10:22, :] = 200
    img[40:52, 34:46, :] = 120
    img[25:28, :, :] = 80
    return img


def _shift_image_zero_fill(img: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape[:2]

    y0_src = max(0, -dy)
    y1_src = min(h, h - dy) if dy >= 0 else h
    y0_dst = max(0, dy)
    y1_dst = y0_dst + (y1_src - y0_src)

    x0_src = max(0, -dx)
    x1_src = min(w, w - dx) if dx >= 0 else w
    x0_dst = max(0, dx)
    x1_dst = x0_dst + (x1_src - x0_src)

    if (y1_src - y0_src) <= 0 or (x1_src - x0_src) <= 0:
        return out

    out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out


def _make_anomaly(img_u8: np.ndarray) -> np.ndarray:
    out = np.array(img_u8, copy=True)
    out[30:42, 18:30] = 255 - out[30:42, 18:30]
    return out


def test_phase_correlation_map_detector_is_misalignment_tolerant() -> None:
    from pyimgano.models.registry import create_model

    base = _make_pattern_image(64)
    shifted = _shift_image_zero_fill(base, dy=5, dx=-7)
    anomaly = _make_anomaly(base)

    det = create_model(
        "vision_phase_correlation_map",
        contamination=0.5,
        n_templates=1,
        resize_hw=(64, 64),
        reduction="mean",
        topk=0.01,
        random_state=0,
    )
    det.fit([base])

    scores = np.asarray(det.decision_function([base, shifted, anomaly]), dtype=np.float64).reshape(-1)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert float(scores[2]) >= float(scores[0])
    assert float(scores[1]) <= float(scores[2])

    maps = det.predict_anomaly_map([base, shifted, anomaly])
    assert maps.shape == (3, 64, 64)
    assert maps.dtype == np.float32
    assert float(np.min(maps)) >= 0.0
    assert float(np.max(maps)) <= 1.0 + 1e-6

    # The shifted image should mostly align in the center (border fill may be noisy).
    margin = 8
    center_mean = float(np.mean(maps[1, margin:-margin, margin:-margin]))
    assert center_mean < 0.25
