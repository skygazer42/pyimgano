import numpy as np


def test_core_hst_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 5))

    det = create_model("core_hst", contamination=0.1, n_trees=9, max_depth=6, random_state=0)
    det.fit(X)
    scores = det.decision_function(X[:13])
    preds = det.predict(X[:13])

    assert scores.shape == (13,)
    assert preds.shape == (13,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_hst_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 4))
    det = create_model(
        "vision_hst",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_trees=5,
        max_depth=5,
        random_state=1,
    )
    det.fit(X)
    scores = det.decision_function(X[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))

