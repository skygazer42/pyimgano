from __future__ import annotations

import numpy as np
import pytest


class _DummyGridMeanEmbedder:
    def __init__(self, *, grid: int = 8) -> None:
        self.grid = int(grid)

    def embed(self, image):  # noqa: ANN001, ANN201 - test stub
        arr = np.asarray(image, dtype=np.uint8)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        g = int(self.grid)
        ph = max(1, h // g)
        pw = max(1, w // g)
        gh = max(1, h // ph)
        gw = max(1, w // pw)

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


def _make_image(*, size: int = 64, value: int = 90) -> np.ndarray:
    return (np.ones((size, size, 3), dtype=np.uint8) * int(value)).astype(np.uint8)


def test_patchcore_lite_map_to_defects_extract_e2e() -> None:
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.defects.extract import extract_defects_from_anomaly_map
    from pyimgano.models.registry import create_model

    embedder = _DummyGridMeanEmbedder(grid=8)
    train = [_make_image(value=90) for _ in range(8)]

    det = create_model(
        "vision_patchcore_lite_map",
        embedder=embedder,
        contamination=0.1,
        n_neighbors=1,
        coreset_sampling_ratio=1.0,
        aggregation_method="topk_mean",
        aggregation_topk=0.05,
    )
    det.fit(train)

    img = _make_image(value=90)
    img[18:46, 10:30, :] = 250
    amap = np.asarray(det.get_anomaly_map(img), dtype=np.float32)
    assert amap.shape == (64, 64)
    assert np.all(np.isfinite(amap))

    defects = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=float(np.quantile(amap, 0.98)),
        roi_xyxy_norm=None,
        mask_space="full",
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=8,
        max_regions=None,
    )
    mask = np.asarray(defects["mask"], dtype=np.uint8)
    assert mask.shape == (64, 64)
    assert int(np.sum(mask > 0)) > 0
