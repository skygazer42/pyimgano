"""Score ensemble example (production pattern).

This script demonstrates a common industrial pattern:
- run multiple detectors
- rank-normalize their scores
- combine into a single robust score

It uses injected embedders so it does not download model weights by default.
"""

from __future__ import annotations

import numpy as np

from pyimgano.inference import infer
from pyimgano.inputs import ImageFormat
from pyimgano.models import create_model


class TinyEmbedder:
    """A tiny patch embedder for demos/tests.

    Returns 4 patch embeddings (2x2 grid). The embedding changes if the image
    contains a bright pixel.
    """

    def embed(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image)}")
        h, w = int(image.shape[0]), int(image.shape[1])
        is_anomaly = bool(image.sum() > 0)
        if is_anomaly:
            patches = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 1))
        else:
            patches = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
        return patches, (2, 2), (h, w)


def main() -> None:
    embedder = TinyEmbedder()

    det_a = create_model(
        "vision_anomalydino",
        embedder=embedder,
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
        coreset_sampling_ratio=1.0,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )
    det_b = create_model(
        "vision_softpatch",
        embedder=embedder,
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
        coreset_sampling_ratio=1.0,
        train_patch_outlier_quantile=0.0,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )

    ensemble = create_model("vision_score_ensemble", detectors=[det_a, det_b], contamination=0.1)

    normal = np.zeros((32, 32, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[0, 0, 0] = 255

    ensemble.fit([normal, normal])
    results = infer(ensemble, [normal, anomaly], input_format=ImageFormat.RGB_U8_HWC)
    for idx, r in enumerate(results):
        print(f"[{idx}] score={r.score:.4f} label={r.label}")


if __name__ == "__main__":
    main()
