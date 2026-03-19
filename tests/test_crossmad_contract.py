from __future__ import annotations

import numpy as np


def test_core_crossmad_fit_predict_smoke() -> None:
    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8)).astype(np.float64)

    det = create_model("core_crossmad", contamination=0.2, num_prototypes=5, random_state=0)
    det.fit(x)
    scores = np.asarray(det.decision_function(x[:10]), dtype=np.float64).reshape(-1)
    preds = np.asarray(det.predict(x[:10]), dtype=np.int64).reshape(-1)

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_crossmad_can_run_on_vectors_with_identity_extractor() -> None:
    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(1)
    x = [rng.normal(size=(12,)).astype(np.float32) for _ in range(60)]

    det = create_model(
        "vision_crossmad",
        contamination=0.25,
        feature_extractor="identity",
        num_prototypes=6,
        random_state=0,
    )
    det.fit(x)
    scores = np.asarray(det.decision_function(x[:5]), dtype=np.float64).reshape(-1)
    preds = np.asarray(det.predict(x[:5]), dtype=np.int64).reshape(-1)

    assert scores.shape == (5,)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
