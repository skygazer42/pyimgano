import numpy as np


def test_core_dbscan_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    # Two tight clusters + some noise
    c1 = rng.normal(loc=0.0, scale=0.2, size=(40, 3))
    c2 = rng.normal(loc=3.0, scale=0.2, size=(40, 3))
    noise = rng.normal(loc=10.0, scale=1.0, size=(5, 3))
    x = np.vstack([c1, c2, noise])

    det = create_model("core_dbscan", contamination=0.1, eps=0.6, min_samples=5)
    det.fit(x)
    scores = det.decision_function(x[:12])
    preds = det.predict(x[:12])

    assert scores.shape == (12,)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
