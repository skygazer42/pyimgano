import numpy as np


def test_core_rrcf_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(64, 4))

    det = create_model("core_rrcf", contamination=0.1, n_trees=7, max_samples=32, random_state=0)
    det.fit(X)
    scores = det.decision_function(X[:9])
    preds = det.predict(X[:9])

    assert scores.shape == (9,)
    assert preds.shape == (9,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_rrcf_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 3))
    det = create_model(
        "vision_rrcf",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_trees=5,
        max_samples=40,
        random_state=1,
    )
    det.fit(X)
    scores = det.decision_function(X[:6])
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))
