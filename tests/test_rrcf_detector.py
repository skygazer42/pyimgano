import numpy as np


def test_rrcf_collusive_displacement_uses_sibling_mass_ratio() -> None:
    from pyimgano.models.rrcf import _collusive_displacement, _RCTNode

    left = _RCTNode(size=1, sample_indices=(0,))
    right = _RCTNode(size=3, sample_indices=(1, 2, 3))
    root = _RCTNode(size=4, left=left, right=right)
    left.parent = root
    right.parent = root

    assert _collusive_displacement(left) == 3.0
    assert _collusive_displacement(right) == 1.0 / 3.0


def test_core_rrcf_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4))

    det = create_model("core_rrcf", contamination=0.1, n_trees=7, max_samples=32, random_state=0)
    det.fit(x)
    scores = det.decision_function(x[:9])
    preds = det.predict(x[:9])

    assert scores.shape == (9,)
    assert preds.shape == (9,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})

    near, far = det.decision_function(np.asarray([[0.0] * 4, [50.0] * 4]))
    assert far > near


def test_vision_rrcf_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    det = create_model(
        "vision_rrcf",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_trees=5,
        max_samples=40,
        random_state=1,
    )
    det.fit(x)
    scores = det.decision_function(x[:6])
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))
