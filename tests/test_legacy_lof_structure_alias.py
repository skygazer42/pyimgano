import numpy as np


def test_legacy_lof_structure_is_still_registered_and_fittable() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4))

    det = create_model(
        "lof_structure",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        n_neighbors=10,
    )
    det.fit(x)
    scores = det.decision_function(x[:9])
    assert scores.shape == (9,)
    assert np.all(np.isfinite(scores))
