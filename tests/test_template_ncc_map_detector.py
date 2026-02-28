from __future__ import annotations

import numpy as np


def _make_gradient_image(size: int = 64) -> np.ndarray:
    x = np.linspace(0.0, 255.0, num=size, dtype=np.float32)
    g = np.tile(x.reshape(1, -1), (size, 1))
    img = np.stack([g, g, g], axis=-1)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _make_anomaly(img_u8: np.ndarray) -> np.ndarray:
    out = np.array(img_u8, copy=True)
    h, w = out.shape[:2]
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 3, 2 * w // 3
    out[y0:y1, x0:x1] = 255 - out[y0:y1, x0:x1]
    return out


def test_template_ncc_map_detector_smoke_and_map_shape() -> None:
    from pyimgano.models.registry import create_model

    base = _make_gradient_image(64)
    anomaly = _make_anomaly(base)

    det = create_model(
        "vision_template_ncc_map",
        contamination=0.5,
        n_templates=1,
        resize_hw=(64, 64),
        window_hw=(9, 9),
        reduction="max",
        topk=0.01,
        random_state=0,
    )
    det.fit([base])

    scores = np.asarray(det.decision_function([base, anomaly]), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert float(scores[1]) >= float(scores[0])

    maps = det.predict_anomaly_map([base, anomaly])
    assert isinstance(maps, np.ndarray)
    assert maps.shape == (2, 64, 64)
    assert maps.dtype == np.float32
    assert float(np.min(maps)) >= 0.0
    assert float(np.max(maps)) <= 1.0 + 1e-6

