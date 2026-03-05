import numpy as np


def test_core_odin_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 5))

    det = create_model("core_odin", contamination=0.1, n_neighbors=10)
    det.fit(X)
    scores = det.decision_function(X[:12])
    preds = det.predict(X[:12])

    assert scores.shape == (12,)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_odin_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(1)
    X = rng.normal(size=(60, 4))
    det = create_model(
        "vision_odin",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_neighbors=7,
    )
    det.fit(X)
    scores = det.decision_function(X[:6])
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))
