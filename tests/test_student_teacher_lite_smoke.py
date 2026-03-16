from __future__ import annotations

import numpy as np


def test_student_teacher_lite_smoke_on_vectors() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = [rng.normal(size=(20,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_student_teacher_lite",
        contamination=0.1,
        teacher_extractor="identity",
        student_extractor={
            "name": "pca_projector",
            "kwargs": {"base_extractor": "identity", "n_components": 0.8},
        },
        ridge=1e-6,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
