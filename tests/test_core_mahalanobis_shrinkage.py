from __future__ import annotations

import numpy as np


def test_core_mahalanobis_shrinkage_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 10)).astype(np.float32)

    det = create_model("core_mahalanobis_shrinkage", contamination=0.1)
    det.fit(x)
    scores = np.asarray(det.decision_function(x[:11]), dtype=np.float64).reshape(-1)
    preds = np.asarray(det.predict(x[:11]), dtype=int).reshape(-1)

    assert scores.shape == (11,)
    assert preds.shape == (11,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
