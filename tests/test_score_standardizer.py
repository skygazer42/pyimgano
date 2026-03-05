from __future__ import annotations

import numpy as np


def test_score_standardizer_rank_in_0_1() -> None:
    from pyimgano.calibration.score_standardization import ScoreStandardizer

    train = np.asarray([10.0, 0.0, 5.0, 5.0])
    std = ScoreStandardizer(method="rank").fit(train)

    out = std.transform(np.asarray([-1.0, 0.0, 5.0, 6.0, 10.0, 11.0]))
    assert out.shape == (6,)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_score_standardizer_zscore_basic() -> None:
    from pyimgano.calibration.score_standardization import ScoreStandardizer

    train = np.asarray([0.0, 1.0, 2.0, 3.0])
    std = ScoreStandardizer(method="zscore").fit(train)
    out = std.transform(train)
    assert out.shape == (4,)
    assert np.isfinite(out).all()
    # mean approx 0
    assert abs(float(np.mean(out))) < 1e-9


def test_score_standardizer_robust_zscore_basic() -> None:
    from pyimgano.calibration.score_standardization import ScoreStandardizer

    train = np.asarray([0.0, 1.0, 2.0, 100.0])  # outlier
    std = ScoreStandardizer(method="robust_zscore").fit(train)
    out = std.transform(train)
    assert out.shape == (4,)
    assert np.isfinite(out).all()


def test_score_standardizer_minmax_clips() -> None:
    from pyimgano.calibration.score_standardization import ScoreStandardizer

    train = np.asarray([2.0, 4.0])
    std = ScoreStandardizer(method="minmax").fit(train)
    out = std.transform(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out.shape == (5,)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
