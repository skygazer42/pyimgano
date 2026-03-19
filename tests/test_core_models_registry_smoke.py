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

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 8))

    try:
        det = create_model(model_name, contamination=0.1)
    except ImportError as exc:
        # Some core_* classical models are gated behind optional extras (e.g. numba).
        # In the minimal test environment (no heavy extras), skip those entries.
        if "pyimgano[" in str(exc) or "Optional dependency" in str(exc):
            pytest.skip(str(exc))
        raise
    det.fit(x)
    scores = det.decision_function(x[:10])
    preds = det.predict(x[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
