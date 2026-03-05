import numpy as np


def test_core_ldof_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 6))

    det = create_model("core_ldof", contamination=0.1, n_neighbors=10)
    det.fit(X)
    scores = det.decision_function(X[:11])
    preds = det.predict(X[:11])

    assert scores.shape == (11,)
    assert preds.shape == (11,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_ldof_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(1)
    X = rng.normal(size=(60, 3))
    det = create_model(
        "vision_ldof",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_neighbors=7,
    )
    det.fit(X)
    scores = det.decision_function(X[:5])
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
