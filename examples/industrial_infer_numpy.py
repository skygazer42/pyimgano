"""Industrial inference example (numpy-first).

This example is self-contained and avoids weight downloads by default
(`pretrained=False`). It demonstrates:

- explicit `ImageFormat`
- canonicalization to `RGB/u8/HWC`
- `infer(..., include_maps=True)` returning scores + labels + anomaly maps
"""

from __future__ import annotations

import numpy as np

from pyimgano.inference import calibrate_threshold, infer
from pyimgano.inputs import ImageFormat, normalize_numpy_image
from pyimgano.models import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess


def _make_normal_bgr(h: int, w: int) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 128


def _make_anomaly_bgr(h: int, w: int) -> np.ndarray:
    img = _make_normal_bgr(h, w)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    return img


def main() -> None:
    # Simulate OpenCV frames: BGR/u8/HWC.
    train_bgr = [_make_normal_bgr(64, 64) for _ in range(8)]
    test_bgr = [_make_normal_bgr(64, 64), _make_anomaly_bgr(64, 64)]

    # Most detectors still need training. For training, pass canonical RGB.
    train_rgb = [
        normalize_numpy_image(frame, input_format=ImageFormat.BGR_U8_HWC) for frame in train_bgr
    ]

    detector = create_model(
        "vision_padim",
        pretrained=False,  # avoids torchvision weight downloads
        device="cpu",
        image_size=64,
        d_reduced=8,
        projection_fit_samples=1,
        covariance_eps=0.1,
    )
    detector.fit(train_rgb)

    # Optional: calibrate a stricter threshold on normal frames (BGR is OK here;
    # `calibrate_threshold` will normalize before calling decision_function).
    calibrate_threshold(detector, train_bgr, input_format=ImageFormat.BGR_U8_HWC, quantile=0.995)

    post = AnomalyMapPostprocess(normalize=True, normalize_method="minmax", gaussian_sigma=1.0)
    results = infer(
        detector,
        test_bgr,
        input_format=ImageFormat.BGR_U8_HWC,
        include_maps=True,
        postprocess=post,
    )

    for idx, r in enumerate(results):
        print(
            f"[{idx}] score={r.score:.4f} label={r.label} "
            f"map={None if r.anomaly_map is None else r.anomaly_map.shape}"
        )


if __name__ == "__main__":
    main()
