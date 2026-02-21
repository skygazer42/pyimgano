import numpy as np
import pytest


class _FakeInferencerMismatchedMaps:
    def predict(self, image_path: str):
        if image_path == "a.png":
            return {"pred_score": 0.1, "anomaly_map": np.zeros((3, 5), dtype=np.float32)}
        if image_path == "b.png":
            return {"pred_score": 0.2, "anomaly_map": np.zeros((4, 5), dtype=np.float32)}
        raise KeyError(image_path)


class _FakeEmbedderMismatchedSizes:
    def embed(self, image_path: str):
        grid_h, grid_w = 2, 2
        patch_embeddings = np.zeros((grid_h * grid_w, 2), dtype=np.float32)
        if image_path == "a.png":
            original_size = (8, 8)
        elif image_path == "b.png":
            original_size = (8, 10)
        else:
            raise KeyError(image_path)
        return patch_embeddings, (grid_h, grid_w), original_size


def test_contamination_is_validated_for_backends():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint
    from pyimgano.models.anomalydino import VisionAnomalyDINO

    inferencer = _FakeInferencerMismatchedMaps()
    with pytest.raises(ValueError):
        VisionAnomalibCheckpoint(
            checkpoint_path="ignored.ckpt",
            inferencer=inferencer,
            contamination=0.0,
        )
    with pytest.raises(ValueError):
        VisionAnomalibCheckpoint(
            checkpoint_path="ignored.ckpt",
            inferencer=inferencer,
            contamination=0.5,
        )

    embedder = _FakeEmbedderMismatchedSizes()
    with pytest.raises(ValueError):
        VisionAnomalyDINO(
            embedder=embedder,
            contamination=0.0,
        )
    with pytest.raises(ValueError):
        VisionAnomalyDINO(
            embedder=embedder,
            contamination=0.5,
        )


def test_empty_training_set_raises_clean_error():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint
    from pyimgano.models.anomalydino import VisionAnomalyDINO

    inferencer = _FakeInferencerMismatchedMaps()
    model_a = VisionAnomalibCheckpoint(
        checkpoint_path="ignored.ckpt",
        inferencer=inferencer,
        contamination=0.1,
    )
    with pytest.raises(ValueError):
        model_a.fit([])

    embedder = _FakeEmbedderMismatchedSizes()
    model_b = VisionAnomalyDINO(
        embedder=embedder,
        contamination=0.1,
    )
    with pytest.raises(ValueError):
        model_b.fit([])


def test_predict_anomaly_map_requires_consistent_shapes():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint
    from pyimgano.models.anomalydino import VisionAnomalyDINO
    from pyimgano.models.openclip_backend import VisionOpenCLIPPromptScore

    inferencer = _FakeInferencerMismatchedMaps()
    model_a = VisionAnomalibCheckpoint(
        checkpoint_path="ignored.ckpt",
        inferencer=inferencer,
        contamination=0.1,
    )
    with pytest.raises(ValueError):
        model_a.predict_anomaly_map(["a.png", "b.png"])

    embedder = _FakeEmbedderMismatchedSizes()
    model_b = VisionAnomalyDINO(
        embedder=embedder,
        contamination=0.1,
    )
    model_b.fit(["a.png"])  # build memory bank
    with pytest.raises(ValueError):
        model_b.predict_anomaly_map(["a.png", "b.png"])

    model_c = VisionOpenCLIPPromptScore(
        embedder=embedder,
        text_features_normal=np.array([1.0, 0.0], dtype=np.float32),
        text_features_anomaly=np.array([0.0, 1.0], dtype=np.float32),
        contamination=0.1,
    )
    model_c.fit(["a.png"])
    with pytest.raises(ValueError):
        model_c.predict_anomaly_map(["a.png", "b.png"])


def test_mvtec_pipeline_resizes_maps_to_mask_shape():
    from pyimgano.pipelines.mvtec_visa import _compute_pixel_scores_from_detector

    class _Detector:
        def get_anomaly_map(self, _path: str):
            return np.zeros((2, 2), dtype=np.float32)

    detector = _Detector()
    paths = ["x.png", "y.png"]
    masks = np.zeros((2, 3, 4), dtype=np.uint8)

    pixel_scores = _compute_pixel_scores_from_detector(detector, paths, masks)
    assert pixel_scores.shape == (2, 3, 4)
