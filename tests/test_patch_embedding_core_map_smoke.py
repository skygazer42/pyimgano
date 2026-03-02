from __future__ import annotations

import numpy as np
import pytest


class _DummyGridMeanEmbedder:
    def __init__(self, *, grid: int = 8) -> None:
        self.grid = int(grid)

    def embed(self, image):  # noqa: ANN001, ANN201 - test stub
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) uint8 image, got {arr.shape} {arr.dtype}")

        h, w = int(arr.shape[0]), int(arr.shape[1])
        g = int(self.grid)
        ph = max(1, h // g)
        pw = max(1, w // g)
        gh = max(1, h // ph)
        gw = max(1, w // pw)

        # Simple patch embedding: mean RGB per patch in [0,1].
        patches: list[np.ndarray] = []
        for iy in range(gh):
            for ix in range(gw):
                y0 = iy * ph
                x0 = ix * pw
                crop = arr[y0 : y0 + ph, x0 : x0 + pw, :].astype(np.float32)
                emb = crop.mean(axis=(0, 1)) / 255.0
                patches.append(emb.reshape(1, 3))

        patch_embeddings = np.concatenate(patches, axis=0).astype(np.float32, copy=False)
        return patch_embeddings, (gh, gw), (h, w)


def _make_image(*, size: int = 64, value: int = 80) -> np.ndarray:
    return (np.ones((size, size, 3), dtype=np.uint8) * int(value)).astype(np.uint8)


def test_patch_embedding_core_map_smoke_predict_and_maps() -> None:
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model

    embedder = _DummyGridMeanEmbedder(grid=8)
    train = [_make_image(value=80) for _ in range(6)]

    det = create_model(
        "vision_patch_embedding_core_map",
        embedder=embedder,
        contamination=0.2,
        core_detector="core_dtc",
        aggregation_method="topk_mean",
        aggregation_topk=0.05,
    )
    det.fit(train)

    normal = _make_image(value=80)
    anomaly = _make_image(value=80)
    anomaly[20:36, 28:44, :] = 250  # bright square defect

    scores = np.asarray(det.decision_function([normal, anomaly]), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert float(scores[1]) > float(scores[0])

    maps = np.asarray(det.predict_anomaly_map([normal, anomaly]), dtype=np.float32)
    assert maps.shape == (2, 64, 64)
    assert np.all(np.isfinite(maps))

