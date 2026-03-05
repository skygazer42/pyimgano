import numpy as np


def test_legacy_dbscan_anomaly_is_still_registered_and_fittable() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(5)
    X = rng.normal(size=(80, 4))

    det = create_model(
        "dbscan_anomaly",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        eps=1.5,
        min_samples=5,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
