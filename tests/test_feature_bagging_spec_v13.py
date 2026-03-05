from __future__ import annotations

import numpy as np


class _IdentityExtractor:
    def extract(self, X):  # noqa: ANN001, ANN201 - test helper
        return np.asarray(X)


def test_core_feature_bagging_spec_register_and_smoke() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 12)).astype(np.float64)

    det = create_model(
        "core_feature_bagging_spec",
        contamination=0.1,
        n_estimators=5,
        max_features=0.75,
        random_state=0,
        base_estimator_spec="core_lof",
    )
    det.fit(X)
    scores = np.asarray(det.decision_function(X), dtype=np.float64).reshape(-1)

    assert scores.shape == (X.shape[0],)
    assert np.all(np.isfinite(scores))


def test_core_feature_bagging_spec_accepts_dict_spec() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 8)).astype(np.float64)

    det = create_model(
        "core_feature_bagging_spec",
        contamination=0.2,
        n_estimators=3,
        max_features=1.0,
        random_state=123,
        base_estimator_spec={"name": "core_lof", "kwargs": {"n_neighbors": 7}},
    )
    det.fit(X)
    scores = np.asarray(det.decision_function(X), dtype=np.float64).reshape(-1)
    assert scores.shape == (X.shape[0],)


def test_vision_feature_bagging_spec_smoke_with_identity_extractor() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(2)
    X = rng.normal(size=(24, 10)).astype(np.float64)
    items = [X[i] for i in range(X.shape[0])]

    det = create_model(
        "vision_feature_bagging_spec",
        feature_extractor=_IdentityExtractor(),
        contamination=0.1,
        n_estimators=4,
        max_features=0.6,
        random_state=0,
        base_estimator_spec="core_lof",
    )
    det.fit(items)
    scores = np.asarray(det.decision_function(items), dtype=np.float64).reshape(-1)
    assert scores.shape == (len(items),)
