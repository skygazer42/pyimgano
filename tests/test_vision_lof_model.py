import numpy as np


def test_vision_lof_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 5))

    det = create_model(
        "vision_lof",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_neighbors=12,
    )
    det.fit(x)
    scores = det.decision_function(x[:7])
    preds = det.predict(x[:7])

    assert scores.shape == (7,)
    assert preds.shape == (7,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
