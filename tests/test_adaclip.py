from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeClipBackend:
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.float32).reshape(-1)


class _FakePromptBackend:
    def get_static_prompts(self):
        return {
            "normal": np.array([1.0, 0.0], dtype=np.float32),
            "anomaly": np.array([0.0, 1.0], dtype=np.float32),
        }

    def adapt(self, support_features: np.ndarray):
        _ = np.asarray(support_features, dtype=np.float32)
        return {
            "normal": np.array([1.0, 0.0], dtype=np.float32),
            "anomaly": np.array([-1.0, 1.0], dtype=np.float32),
        }


def test_adaclip_hybrid_prompt_fusion_changes_score_and_ranks_anomaly_higher() -> None:
    support = [np.array([1.0, 0.0], dtype=np.float32)]
    normal = np.array([1.0, 0.0], dtype=np.float32)
    anomaly = np.array([0.0, 2.0], dtype=np.float32)

    static_only = models.create_model(
        "vision_adaclip",
        clip_backend=_FakeClipBackend(),
        prompt_backend=_FakePromptBackend(),
        dynamic_prompt_weight=0.0,
    )
    static_only.fit(support)

    hybrid = models.create_model(
        "vision_adaclip",
        clip_backend=_FakeClipBackend(),
        prompt_backend=_FakePromptBackend(),
        dynamic_prompt_weight=0.6,
    )
    hybrid.fit(support)

    static_scores = np.asarray(static_only.decision_function([normal, anomaly]), dtype=np.float64)
    hybrid_scores = np.asarray(hybrid.decision_function([normal, anomaly]), dtype=np.float64)

    assert static_scores.shape == (2,)
    assert hybrid_scores.shape == (2,)
    assert float(hybrid_scores[1]) > float(hybrid_scores[0])
    assert not np.isclose(float(hybrid_scores[1]), float(static_scores[1]))
