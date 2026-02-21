import numpy as np

from pyimgano.models import create_model


class _FakePatchEmbedder:
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


def test_softpatch_registry_and_basic_api():
    det = create_model(
        "vision_softpatch",
        embedder=_FakePatchEmbedder(),
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )

    det.fit(["train_1.png", "train_2.png"])

    scores = det.decision_function(["normal.png", "anomaly.png"])
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

    anomaly_map = det.get_anomaly_map("anomaly.png")
    assert anomaly_map.shape == (8, 8)
    assert np.isfinite(anomaly_map).all()

