from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeFeatureExtractor:
    def __call__(self, image: np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        return {
            "low": arr.reshape(-1),
            "high": (arr * 2.0).reshape(-1),
        }


def test_univad_fuses_multilayer_features_and_scores_shift_higher() -> None:
    detector = models.create_model(
        "vision_univad",
        feature_extractor=_FakeFeatureExtractor(),
        layer_weights={"low": 0.25, "high": 0.75},
        contamination=0.25,
    )

    normal = np.array([[0.0, 0.1], [0.0, 0.1]], dtype=np.float32)
    shifted = np.array([[3.0, 3.1], [3.0, 3.1]], dtype=np.float32)

    detector.fit([normal, normal + 0.05])
    scores = np.asarray(detector.decision_function([normal, shifted]), dtype=np.float64)
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])
