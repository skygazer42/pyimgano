import numpy as np

from pyimgano.models import create_model


class _ArrayEmbedder:
    def embed(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image)}")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB/u8/HWC, got {image.shape} {image.dtype}")

        h, w = int(image.shape[0]), int(image.shape[1])
        is_anomaly = bool(image.sum() > 0)
        if is_anomaly:
            patches = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 1))
        else:
            patches = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
        return patches, (2, 2), (h, w)


def test_softpatch_accepts_numpy_images_for_fit_scoring_and_maps():
    embedder = _ArrayEmbedder()
    det = create_model(
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

    normal = np.zeros((10, 12, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[0, 0, 0] = 1

    det.fit([normal, normal])

    scores = det.decision_function([normal, anomaly])
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

    anomaly_map = det.get_anomaly_map(anomaly)
    assert anomaly_map.shape == (10, 12)
    assert anomaly_map.dtype == np.float32
    assert np.isfinite(anomaly_map).all()
