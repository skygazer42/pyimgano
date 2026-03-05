from __future__ import annotations

import numpy as np


def test_minmax_in_0_1() -> None:
    from pyimgano.utils.score_normalization import minmax

    x = np.array([2.0, 4.0, 3.0])
    y = minmax(x)
    assert y.shape == (3,)
    assert np.isfinite(y).all()
    assert 0.0 <= float(y.min()) <= 1.0
    assert 0.0 <= float(y.max()) <= 1.0


def test_rank01_in_0_1_and_monotone() -> None:
    from pyimgano.utils.score_normalization import rank01

    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = rank01(x)
    assert y.shape == (4,)
    assert np.isfinite(y).all()
    assert float(y.min()) == 0.0
    assert float(y.max()) == 1.0
    assert np.all(np.diff(y) >= 0)


def test_rank01_ties_average_rank() -> None:
    from pyimgano.utils.score_normalization import rank01

    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = rank01(x)
    # The first two entries tie and should get the same rank.
    assert float(y[0]) == float(y[1])


def test_zscore_finite() -> None:
    from pyimgano.utils.score_normalization import zscore

    x = np.array([1.0, 2.0, 3.0])
    y = zscore(x)
    assert y.shape == (3,)
    assert np.isfinite(y).all()


def test_normalize_dispatch() -> None:
    from pyimgano.utils.score_normalization import normalize

    x = np.array([1.0, 2.0, 3.0])
    assert normalize(x, "minmax").shape == (3,)
    assert normalize(x, "rank").shape == (3,)
    assert normalize(x, "quantile").shape == (3,)
    assert normalize(x, "zscore").shape == (3,)
