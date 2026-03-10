from __future__ import annotations

import numpy as np

import pyimgano.models as models


class FakePatchSearchBackend:
    def fit(self, train_patches):
        self.train_centroid = np.mean(np.concatenate(train_patches, axis=0), axis=0)
        return self

    def score(self, patch_grid):
        delta = patch_grid - self.train_centroid[None, :]
        patch_scores = np.linalg.norm(delta, axis=1)
        return float(np.max(patch_scores)), patch_scores


def test_visionad_scores_anomaly_higher_than_normal() -> None:
    detector = models.create_model(
        "vision_visionad",
        search_backend=FakePatchSearchBackend(),
        embedder=lambda image: image,
        contamination=0.25,
    )
    train = [np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)]
    test = [
        np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32),
        np.array([[5.0, 5.0], [5.1, 5.1]], dtype=np.float32),
    ]
    detector.fit(train)
    scores = detector.decision_function(test)
    assert float(scores[1]) > float(scores[0])
