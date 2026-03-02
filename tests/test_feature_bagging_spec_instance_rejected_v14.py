from __future__ import annotations

import numpy as np
import pytest


class _DummyEstimator:
    def decision_function(self, X):  # noqa: ANN001, ANN201 - test helper
        return np.zeros(len(np.asarray(X)), dtype=np.float64)


def test_feature_bagging_spec_rejects_estimator_instances() -> None:
    """Spec-friendly ensembles must stay JSON-configurable (no object instances)."""

    from pyimgano.models import create_model

    det = create_model(
        "core_feature_bagging_spec",
        contamination=0.1,
        n_estimators=3,
        max_features=1.0,
        random_state=0,
        base_estimator_spec=_DummyEstimator(),
    )

    X = np.random.default_rng(0).normal(size=(16, 4)).astype(np.float64)
    with pytest.raises(TypeError, match="Instances are not supported"):
        det.fit(X)

