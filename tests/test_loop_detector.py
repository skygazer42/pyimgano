import numpy as np


def test_core_loop_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401 - registry population

    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(64, 5))

    det = create_model("core_loop", contamination=0.1, n_neighbors=10, lambda_=3.0)
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_loop_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 4))
    det = create_model(
        "vision_loop",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_neighbors=5,
    )
    det.fit(X)
    scores = det.decision_function(X[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))

