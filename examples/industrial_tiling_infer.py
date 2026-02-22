"""High-resolution tiling inference example (numpy-first).

This example demonstrates:
- fitting a detector on normal numpy images
- wrapping it with `TiledDetector` for high-res inference
- running `infer(..., include_maps=True)` to get an image score + stitched anomaly map

Notes
-----
- Uses `pretrained=False` to avoid downloading backbone weights in examples.
"""

from __future__ import annotations

import numpy as np

from pyimgano.inference import TiledDetector, calibrate_threshold, infer
from pyimgano.inputs import ImageFormat
from pyimgano.models import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess


def _make_normal_rgb(h: int, w: int) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 128


def _make_anomaly_rgb(h: int, w: int) -> np.ndarray:
    img = _make_normal_rgb(h, w)
    # Tiny defect (hard for aggressive resize-only pipelines)
    img[h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4] = 255
    return img


def main() -> None:
    # "Training" normals at the tile scale.
    train = [_make_normal_rgb(96, 96) for _ in range(8)]

    base = create_model(
        "vision_padim",
        pretrained=False,
        device="cpu",
        image_size=96,
        d_reduced=8,
        projection_fit_samples=1,
        covariance_eps=0.1,
    )

    tiled = TiledDetector(
        detector=base,
        tile_size=96,
        stride=64,
        score_reduce="max",
        map_reduce="hann",  # weighted blending to reduce seam artifacts
    )
    tiled.fit(train)
    calibrate_threshold(tiled, train, input_format=ImageFormat.RGB_U8_HWC, quantile=0.995)

    test = [_make_normal_rgb(256, 256), _make_anomaly_rgb(256, 256)]
    post = AnomalyMapPostprocess(normalize=True, normalize_method="minmax", gaussian_sigma=1.0)

    results = infer(
        tiled,
        test,
        input_format=ImageFormat.RGB_U8_HWC,
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
