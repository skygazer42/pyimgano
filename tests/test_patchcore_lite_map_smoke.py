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


def test_patchcore_lite_map_smoke_predict_and_maps() -> None:
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model

    embedder = _DummyGridMeanEmbedder(grid=8)
    train = [_make_image(value=80) for _ in range(6)]

    det = create_model(
        "vision_patchcore_lite_map",
        embedder=embedder,
        contamination=0.2,
        knn_backend="sklearn",
        metric="euclidean",
        n_neighbors=1,
        coreset_sampling_ratio=1.0,
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


def test_patchcore_lite_map_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from PIL import Image

    from pyimgano.features.torchvision_conv_patch_embedder import TorchvisionConvPatchEmbedder
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(path)
        train_paths.append(str(path))

    normal_path = tmp_path / "normal.png"
    anomaly_path = tmp_path / "anomaly.png"
    Image.fromarray(np.full((32, 32, 3), 83, dtype=np.uint8), mode="RGB").save(normal_path)
    anomaly = np.full((32, 32, 3), 83, dtype=np.uint8)
    anomaly[10:22, 10:22, :] = 240
    Image.fromarray(anomaly, mode="RGB").save(anomaly_path)

    def _make_detector():
        embedder = TorchvisionConvPatchEmbedder(
            backbone="resnet18",
            node="layer3",
            pretrained=False,
            device="cpu",
            image_size=32,
            normalize=True,
        )
        return create_model(
            "vision_patchcore_lite_map",
            embedder=embedder,
            contamination=0.2,
            knn_backend="sklearn",
            metric="euclidean",
            n_neighbors=1,
            coreset_sampling_ratio=1.0,
            aggregation_method="topk_mean",
            aggregation_topk=0.05,
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_maps = np.asarray(
        detector.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "patchcore_lite.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_maps = np.asarray(
        restored.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_maps, expected_maps, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)
