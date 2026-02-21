import numpy as np


def test_anomalydino_aggregate_topk_mean():
    from pyimgano.models.patchknn_core import aggregate_patch_scores

    patch_scores = np.arange(10, dtype=np.float32)
    assert aggregate_patch_scores(patch_scores, method="topk_mean", topk=0.2) == 8.5
    assert aggregate_patch_scores(patch_scores, method="topk_mean", topk=0.01) == 9.0


def test_anomalydino_reshape_patch_scores():
    from pyimgano.models.patchknn_core import reshape_patch_scores

    patch_scores = np.arange(6, dtype=np.float32)
    grid = reshape_patch_scores(patch_scores, grid_h=2, grid_w=3)
    assert grid.shape == (2, 3)
    assert float(grid[0, 0]) == 0.0
    assert float(grid[1, 2]) == 5.0


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


def test_vision_anomalydino_fit_scores_and_anomaly_map():
    from pyimgano.models.anomalydino import VisionAnomalyDINO

    embedder = _FakePatchEmbedder()
    model = VisionAnomalyDINO(
        embedder=embedder,
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
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


def test_vision_anomalydino_coreset_sampling_reduces_memory_bank():
    from pyimgano.models.anomalydino import VisionAnomalyDINO

    class _CountingEmbedder:
        def embed(self, image_path: str):
            grid_h, grid_w = 2, 2
            original_h, original_w = 8, 8
            offset = float(sum(ord(c) for c in image_path) % 10)
            patch_embeddings = (np.arange(grid_h * grid_w, dtype=np.float32).reshape(-1, 1) + offset)
            return patch_embeddings, (grid_h, grid_w), (original_h, original_w)

    embedder = _CountingEmbedder()
    det_full = VisionAnomalyDINO(embedder=embedder, coreset_sampling_ratio=1.0)
    det_full.fit(["a.png", "b.png", "c.png"])

    det_half = VisionAnomalyDINO(embedder=embedder, coreset_sampling_ratio=0.5, random_seed=0)
    det_half.fit(["a.png", "b.png", "c.png"])

    assert det_half.memory_bank_size_ < det_full.memory_bank_size_
