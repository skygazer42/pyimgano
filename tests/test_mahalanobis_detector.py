import numpy as np


def test_core_mahalanobis_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 6))

    det = create_model("core_mahalanobis", contamination=0.1, reg=1e-6)
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_mahalanobis_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 4))
    det = create_model(
        "vision_mahalanobis",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        reg=1e-6,
    )
    det.fit(X)
    scores = det.decision_function(X[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))

