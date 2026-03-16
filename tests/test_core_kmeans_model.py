import numpy as np


def test_core_kmeans_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))

    det = create_model("core_kmeans", contamination=0.1, n_clusters=5, random_state=0)
    det.fit(X)
    scores = det.decision_function(X[:11])
    preds = det.predict(X[:11])

    assert scores.shape == (11,)
    assert preds.shape == (11,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
