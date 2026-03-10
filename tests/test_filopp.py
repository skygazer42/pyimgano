from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeVLMBackend:
    def encode_image(self, image: np.ndarray):
        patches = np.asarray(image, dtype=np.float32)
        return patches.reshape(-1, 2), (2, 2)


class _FakeLocalizationBackend:
    def fit(self, support_patch_sets):
        self.prototype_ = np.mean(np.concatenate(support_patch_sets, axis=0), axis=0)
        return self

    def score(self, patch_embeddings: np.ndarray):
        delta = np.asarray(patch_embeddings, dtype=np.float32) - self.prototype_[None, :]
        patch_scores = np.linalg.norm(delta, axis=1).astype(np.float32)
        return float(np.max(patch_scores)), patch_scores


def test_filopp_localizes_anomalous_patch_and_ranks_anomalous_image_higher() -> None:
    detector = models.create_model(
        "vision_filopp",
        vlm_backend=_FakeVLMBackend(),
        localization_backend=_FakeLocalizationBackend(),
        contamination=0.25,
    )

    support = [
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    ]
    normal = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    anomaly = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32)

    detector.fit(support)

    scores = np.asarray(detector.decision_function([normal, anomaly]), dtype=np.float64)
    maps = np.asarray(detector.predict_anomaly_map([normal, anomaly]), dtype=np.float32)

    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])
    assert maps.shape == (2, 2, 2)
    assert float(np.max(maps[1])) > float(np.max(maps[0]))
