from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize(
    "model_name",
    [
        # Representative "core_*" feature-matrix detectors; keep this list short
        # to avoid exploding test time while still enforcing the contract.
        "core_ecod",
        "core_lof",
        "core_iforest",
        "core_knn_cosine",
        "core_mahalanobis_shrinkage",
    ],
)
def test_core_detector_contract_decision_function_shape_dtype_finite(model_name: str) -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal(size=(128, 16)).astype(np.float32)
    X_test = rng.standard_normal(size=(7, 16)).astype(np.float32)

    detector = create_model(model_name, contamination=0.1)
    fitted = detector.fit(X_train)
    assert fitted is detector

    scores = detector.decision_function(X_test)
    scores = np.asarray(scores)

    assert scores.shape == (X_test.shape[0],)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()
