import numpy as np


def test_legacy_isolation_forest_struct_is_still_registered_and_fittable() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 6))

    det = create_model(
        "isolation_forest_struct",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        n_estimators=25,
        random_state=3,
    )
    det.fit(x)
    scores = det.decision_function(x[:8])
    assert scores.shape == (8,)
    assert np.all(np.isfinite(scores))
