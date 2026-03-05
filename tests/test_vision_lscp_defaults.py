from __future__ import annotations

import numpy as np


def _toy_vectors(seed: int = 0, *, n: int = 80, d: int = 8) -> list[np.ndarray]:
    rng = np.random.RandomState(int(seed))
    return [rng.normal(size=(int(d),)).astype(np.float32) for _ in range(int(n))]


def test_vision_lscp_default_detectors_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.models import create_model

    X = _toy_vectors(0, n=80, d=8)

    det = create_model(
        "vision_lscp",
        contamination=0.1,
        feature_extractor="identity",
        random_state=0,
        local_region_iterations=5,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_lscp_spec_default_detectors_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.models import create_model

    X = _toy_vectors(1, n=80, d=8)

    det = create_model(
        "vision_lscp_spec",
        contamination=0.1,
        feature_extractor="identity",
        random_state=0,
        local_region_iterations=5,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
