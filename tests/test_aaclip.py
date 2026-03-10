from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeClipBackend:
    def encode_image(self, image: np.ndarray):
        patches = np.asarray(image, dtype=np.float32)
        return patches.reshape(-1, 2), (2, 2)


class _FakeAnchorBackend:
    def fit(self, support_patch_sets):
        support = np.concatenate(support_patch_sets, axis=0)
        self.normal_anchor_ = np.mean(support, axis=0).astype(np.float32)
        self.anomaly_anchor_ = np.array([0.0, 1.0], dtype=np.float32)
        return self

    def get_anchors(self):
        return {
            "normal": self.normal_anchor_,
            "anomaly": self.anomaly_anchor_,
        }


def test_aaclip_patchwise_anchor_scoring_highlights_anomalies() -> None:
    detector = models.create_model(
        "vision_aaclip",
        clip_backend=_FakeClipBackend(),
        anchor_backend=_FakeAnchorBackend(),
        contamination=0.25,
    )

    support = [
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    ]
    normal = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    anomaly = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 3.0], [1.0, 0.0]], dtype=np.float32)

    detector.fit(support)

    scores = np.asarray(detector.decision_function([normal, anomaly]), dtype=np.float64)
    maps = np.asarray(detector.predict_anomaly_map([normal, anomaly]), dtype=np.float32)
    labels = np.asarray(detector.predict([normal, anomaly]), dtype=np.int64)

    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])
    assert maps.shape == (2, 2, 2)
    assert float(np.max(maps[1])) > float(np.max(maps[0]))
    assert labels.tolist() == [0, 1]
