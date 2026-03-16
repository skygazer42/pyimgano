from __future__ import annotations

import numpy as np


def test_patchcore_online_smoke_on_vectors() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal(size=(10, 8)).astype(np.float32)
    x_new = rng.standard_normal(size=(5, 8)).astype(np.float32)
    x_test = rng.standard_normal(size=(3, 8)).astype(np.float32)

    det = create_model(
        "vision_patchcore_online",
        feature_extractor="identity",
        contamination=0.2,
        max_bank_size=12,
        random_state=0,
    )

    det.fit(x_train)
    scores1 = np.asarray(det.decision_function(x_test), dtype=np.float64)
    assert scores1.shape == (x_test.shape[0],)
    assert np.isfinite(scores1).all()

    det.partial_fit(x_new)
    bank = getattr(det.detector, "memory_bank_", None)
    assert bank is not None
    assert int(bank.shape[0]) <= 12
    assert int(bank.shape[0]) > 0

    scores2 = np.asarray(det.decision_function(x_test), dtype=np.float64)
    assert scores2.shape == (x_test.shape[0],)
    assert np.isfinite(scores2).all()
