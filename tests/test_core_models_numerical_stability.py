from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize(
    "model_name",
    [
        "core_mst_outlier",
        "core_lid",
        "core_kde_ratio",
        "core_cook_distance",
        "core_studentized_residual",
        "core_extra_trees_density",
        "core_neighborhood_entropy",
    ],
)
def test_selected_core_models_handle_constant_inputs(model_name: str) -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    X = np.zeros((40, 6), dtype=np.float64)
    det = create_model(model_name, contamination=0.1)
    det.fit(X)
    scores = np.asarray(det.decision_function(X[:10]), dtype=np.float64)

    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
