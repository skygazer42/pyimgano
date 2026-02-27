import numpy as np


def test_legacy_kmeans_anomaly_is_still_registered_and_fittable() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.RandomState(4)
    X = rng.normal(size=(70, 5))

    det = create_model(
        "kmeans_anomaly",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        n_clusters=6,
        random_state=4,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))

