from __future__ import annotations

import numpy as np

from pyimgano.models.kpca import CoreKPCA, _PyODKernelPCA


def test_pyod_kpca_exposes_lowercase_copy_x_for_sklearn_params() -> None:
    wrapped = _PyODKernelPCA(copy_x=False)

    params = wrapped.get_params(deep=False)

    assert params["copy_x"] is False
    assert wrapped.copy_X is False


def test_core_kpca_fit_and_score_smoke() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 6))

    detector = CoreKPCA(contamination=0.1, n_components=8, random_state=0)
    detector.fit(X)
    scores = detector.decision_function(X[:5])

    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
