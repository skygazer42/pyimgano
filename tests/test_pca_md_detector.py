import numpy as np


def test_core_pca_md_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 10))

    det = create_model("core_pca_md", contamination=0.1, n_components=5, random_state=0)
    det.fit(x)
    scores = det.decision_function(x[:12])
    preds = det.predict(x[:12])

    assert scores.shape == (12,)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_pca_md_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 8))
    det = create_model(
        "vision_pca_md",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_components=0.9,
        random_state=1,
    )
    det.fit(x)
    scores = det.decision_function(x[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))
