from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeAnoGenerator:
    def generate(self, image: np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        generated = np.array(arr, copy=True)
        generated[1:3, 1:3] += 4.0
        mask = np.zeros(arr.shape[:2], dtype=np.float32)
        mask[1:3, 1:3] = 1.0
        return generated, mask, {"kind": "square"}


def test_anogen_adapter_generates_pairs_and_scores_anomaly_higher() -> None:
    detector = models.create_model(
        "vision_anogen_adapter",
        generator=_FakeAnoGenerator(),
    )

    normal = np.zeros((4, 4), dtype=np.float32)
    shifted = np.zeros((4, 4), dtype=np.float32)
    shifted[1:3, 1:3] = 3.0

    pairs = detector.generate_training_pairs([normal])
    assert len(pairs) == 1
    assert np.allclose(pairs[0]["normal"], normal)
    assert pairs[0]["anomalous"].shape == normal.shape
    assert pairs[0]["mask"].shape == normal.shape
    assert np.isclose(float(np.sum(pairs[0]["mask"])), 4.0)
    assert pairs[0]["meta"]["kind"] == "square"

    detector.fit([normal])
    scores = np.asarray(detector.decision_function([normal, shifted]), dtype=np.float64)
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])
