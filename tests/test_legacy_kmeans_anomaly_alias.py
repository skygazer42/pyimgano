import numpy as np


def test_legacy_kmeans_anomaly_is_still_registered_and_fittable() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(4)
    x = rng.normal(size=(70, 5))

    det = create_model(
        "kmeans_anomaly",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        n_clusters=6,
        random_state=4,
    )
    det.fit(x)
    scores = det.decision_function(x[:10])
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
