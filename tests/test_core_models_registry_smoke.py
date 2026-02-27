from __future__ import annotations

import numpy as np
import pytest


def _core_model_names() -> list[str]:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import MODEL_REGISTRY, list_models

    names = [n for n in list_models() if n.startswith("core_")]
    # Keep this smoke test lightweight: core_* deep models can be heavier and
    # already have dedicated tests.
    out: list[str] = []
    for n in names:
        entry = MODEL_REGISTRY.info(n)
        if "deep" in entry.tags:
            continue
        out.append(n)
    return sorted(out)


@pytest.mark.parametrize("model_name", _core_model_names())
def test_core_models_fit_predict_smoke(model_name: str) -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 8))

    det = create_model(model_name, contamination=0.1)
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})

