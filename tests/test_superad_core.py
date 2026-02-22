import numpy as np


class _FakePatchEmbedder:
    """Deterministic embedder returning (patch_embeddings, grid_shape, original_size)."""

    def embed(self, image_path: str):
        grid_h, grid_w = 2, 2
        original_h, original_w = 8, 8

        if "anomaly" in image_path:
            patch_embeddings = np.array(
                [
                    [10.0, 10.0],
                    [0.0, 0.0],
                    [10.0, 10.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
        else:
            patch_embeddings = np.zeros((grid_h * grid_w, 2), dtype=np.float32)

        return patch_embeddings, (grid_h, grid_w), (original_h, original_w)


def test_vision_superad_fit_scores_and_anomaly_map() -> None:
    from pyimgano.models.superad import VisionSuperAD

    embedder = _FakePatchEmbedder()
    model = VisionSuperAD(
        embedder=embedder,
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=2,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )
    model.fit(["train_1.png", "train_2.png"])

    scores = model.decision_function(["normal.png", "anomaly.png"])
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

    y_pred = model.predict(["normal.png", "anomaly.png"])
    assert y_pred.tolist() == [0, 1]

    normal_map = model.get_anomaly_map("normal.png")
    anomaly_map = model.get_anomaly_map("anomaly.png")
    assert normal_map.shape == (8, 8)
    assert anomaly_map.shape == (8, 8)
    assert np.isfinite(normal_map).all()
    assert np.isfinite(anomaly_map).all()
    assert float(anomaly_map.mean()) > float(normal_map.mean())

