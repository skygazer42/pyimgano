import numpy as np


def test_core_knn_degree_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 5))

    det = create_model("core_knn_degree", contamination=0.1, n_neighbors=10, radius_quantile=0.5)
    det.fit(x)
    scores = det.decision_function(x[:12])
    preds = det.predict(x[:12])

    assert scores.shape == (12,)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_knn_degree_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    det = create_model(
        "vision_knn_degree",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_neighbors=7,
        radius_quantile=0.5,
    )
    det.fit(x)
    scores = det.decision_function(x[:6])
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))
