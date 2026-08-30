from __future__ import annotations

import numpy as np
import pytest

from pyimgano.models import create_model
from pyimgano.models.registry import model_info


class _OfficialBackendStub:
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, items, y=None):  # noqa: ANN001
        del items, y
        self.fitted = True

    def decision_function(self, items, batch_size=None):  # noqa: ANN001
        del batch_size
        return np.asarray([float(np.asarray(item).mean()) for item in items])

    def predict_with_uncertainty(self, items):  # noqa: ANN001
        scores = self.decision_function(items)
        return scores, np.full_like(scores, 0.25)

    def predict_anomaly_map(self, items):  # noqa: ANN001
        return np.stack([np.asarray(item, dtype=np.float32).mean(axis=-1) for item in items])


def test_bayesianpf_requires_official_backend_instead_of_random_fallback() -> None:
    detector = create_model("vision_bayesianpf")
    with pytest.raises(RuntimeError, match="trained official Bayes-PFL backend"):
        detector.fit(np.zeros((1, 4, 4, 3), dtype=np.uint8))


def test_bayesianpf_adapter_preserves_detector_contract() -> None:
    backend = _OfficialBackendStub()
    detector = create_model(
        "vision_bayesianpf",
        backend=backend,
        contamination=0.25,
    )
    images = np.stack(
        [
            np.zeros((4, 4, 3), dtype=np.float32),
            np.ones((4, 4, 3), dtype=np.float32),
            np.full((4, 4, 3), 2.0, dtype=np.float32),
            np.full((4, 4, 3), 3.0, dtype=np.float32),
        ]
    )

    assert detector.fit(images) is detector
    assert backend.fitted is True
    np.testing.assert_allclose(detector.decision_function(images), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(detector.predict(images), [0, 0, 0, 1])
    scores, uncertainty = detector.predict_with_uncertainty(images)
    np.testing.assert_allclose(scores, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(uncertainty, 0.25)
    assert detector.predict_anomaly_map(images).shape == (4, 4, 4)


def test_bayesianpf_registry_discloses_external_checkpoint_requirement() -> None:
    info = model_info("vision_bayesianpf")
    assert info["metadata"]["implementation_status"] == "user-supplied-external-backend-facade"
    assert info["metadata"]["requires_checkpoint"] is True
    assert "sota" not in info["tags"]
