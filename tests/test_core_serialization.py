from __future__ import annotations

import numpy as np


def test_core_model_joblib_roundtrip(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model
    from pyimgano.models.serialization import load_model, save_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 6))

    det = create_model("core_ecod", contamination=0.1)
    det.fit(X)
    s0 = np.asarray(det.decision_function(X[:10]), dtype=np.float64)

    p = save_model(det, tmp_path / "model.joblib")
    det2 = load_model(p)
    s1 = np.asarray(det2.decision_function(X[:10]), dtype=np.float64)

    assert s0.shape == s1.shape
    assert np.all(np.isfinite(s1))
    assert np.allclose(s0, s1, atol=1e-12, rtol=1e-12)

