import numpy as np


def test_core_dcorr_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))

    det = create_model("core_dcorr", contamination=0.1, n_projections=3, random_state=0)
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_dcorr_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    det = create_model(
        "vision_dcorr",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_projections=2,
        random_state=1,
    )
    det.fit(X)
    scores = det.decision_function(X[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))
