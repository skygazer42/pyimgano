import cv2
import numpy as np

from pyimgano.models import create_model


class _FakePatchEmbedder:
    def embed(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        height, width = int(img.shape[0]), int(img.shape[1])

        # 4 patches (2x2), dim=2, deterministic by path.
        if "anomaly" in image_path:
            patches = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 1))
        else:
            patches = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))

        return patches, (2, 2), (height, width)


def test_pixel_anomaly_map_contract_for_key_detectors(tmp_path):
    normal_paths: list[str] = []
    for i in range(2):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        path = tmp_path / f"normal_{i}.png"
        cv2.imwrite(str(path), img)
        normal_paths.append(str(path))

    anomaly_path = tmp_path / "anomaly.png"
    anomaly_img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    anomaly_img[16:48, 16:48] = 255
    cv2.imwrite(str(anomaly_path), anomaly_img)

    embedder = _FakePatchEmbedder()

    detectors = [
        create_model(
            "vision_patchcore",
            coreset_sampling_ratio=1.0,
            pretrained=False,
            device="cpu",
        ),
        create_model(
            "vision_anomalydino",
            embedder=embedder,
            contamination=0.1,
            knn_backend="sklearn",
            n_neighbors=1,
            coreset_sampling_ratio=1.0,
        ),
        create_model(
            "vision_softpatch",
            embedder=embedder,
            contamination=0.1,
            knn_backend="sklearn",
            n_neighbors=1,
            coreset_sampling_ratio=1.0,
            train_patch_outlier_quantile=0.0,
        ),
        create_model(
            "vision_openclip_patchknn",
            embedder=embedder,
            contamination=0.1,
            knn_backend="sklearn",
            n_neighbors=1,
            coreset_sampling_ratio=1.0,
        ),
        create_model(
            "vision_openclip_promptscore",
            embedder=embedder,
            text_features_normal=np.array([1.0, 0.0], dtype=np.float32),
            text_features_anomaly=np.array([0.0, 1.0], dtype=np.float32),
            contamination=0.1,
            aggregation_method="topk_mean",
            aggregation_topk=0.25,
        ),
    ]

    for detector in detectors:
        detector.fit(normal_paths)

        anomaly_map = detector.get_anomaly_map(str(anomaly_path))
        assert isinstance(anomaly_map, np.ndarray)
        assert anomaly_map.dtype == np.float32
        assert anomaly_map.ndim == 2
        assert anomaly_map.shape == (64, 64)
        assert np.isfinite(anomaly_map).all()

