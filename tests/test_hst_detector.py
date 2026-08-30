import numpy as np


def test_hst_uses_midpoint_cuts() -> None:
    from pyimgano.models.hst import _build_hst

    tree = _build_hst(
        np.asarray([0.0]),
        np.asarray([2.0]),
        max_depth=1,
        rng=np.random.default_rng(0),
    )

    assert tree.split_dims[0] == 0
    assert tree.split_values[0] == 1.0


def test_hst_promotes_latest_mass_at_window_boundary() -> None:
    from pyimgano.models.hst import CoreHST

    detector = CoreHST(
        n_trees=1,
        max_depth=2,
        window_size=2,
        size_limit=0,
        assume_normalized=True,
        random_state=0,
    ).fit(np.asarray([[0.1], [0.2], [0.3]]))
    tree = detector._forest[0]
    old_reference = tree.reference_mass.copy()

    scores = detector.update(np.asarray([[0.8], [0.9]]))

    assert scores.shape == (2,)
    assert detector._latest_window_count == 0
    assert np.all(tree.latest_mass == 0)
    assert tree.reference_mass[0] == 2
    assert not np.array_equal(tree.reference_mass, old_reference)


def test_core_hst_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 5))

    det = create_model("core_hst", contamination=0.1, n_trees=9, max_depth=6, random_state=0)
    det.fit(x)
    scores = det.decision_function(x[:13])
    preds = det.predict(x[:13])

    assert scores.shape == (13,)
    assert preds.shape == (13,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_hst_with_identity_extractor() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.models import create_model

    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4))
    det = create_model(
        "vision_hst",
        feature_extractor=IdentityExtractor(),
        contamination=0.2,
        n_trees=5,
        max_depth=5,
        random_state=1,
    )
    det.fit(x)
    scores = det.decision_function(x[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))
