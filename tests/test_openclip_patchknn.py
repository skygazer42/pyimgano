import numpy as np

from pyimgano import models


class _FakeEmbedder:
    def embed(self, image_path: str):
        # 4 patches (2x2), dim=2, deterministic by path.
        base = 0.0 if "normal" in image_path else 10.0
        patches = np.array(
            [
                [base, 0.0],
                [base, 1.0],
                [base, 2.0],
                [base, 3.0],
            ],
            dtype=np.float32,
        )
        return patches, (2, 2), (8, 8)


def test_openclip_patchknn_scores_higher_for_anomaly():
    detector = models.create_model(
        "vision_openclip_patchknn",
        embedder=_FakeEmbedder(),
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )
    detector.fit(["normal_1.png", "normal_2.png"])
    scores = detector.decision_function(["normal_x.png", "anomaly_x.png"])
    assert float(scores[1]) > float(scores[0])
