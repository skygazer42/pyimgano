import numpy as np


def test_core_lof_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))

    det = create_model("core_lof", contamination=0.1, n_neighbors=10)
    det.fit(X)
    scores = det.decision_function(X[:12])
    preds = det.predict(X[:12])

    assert scores.shape == (12,)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
