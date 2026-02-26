import numpy as np
import pytest


def test_rank_calibrator_requires_fit() -> None:
    from pyimgano.calibration.rank_calibration import RankCalibrator

    cal = RankCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.transform([0.0, 1.0])


def test_rank_calibrator_empirical_cdf_mapping() -> None:
    from pyimgano.calibration.rank_calibration import RankCalibrator

    train = np.asarray([0.0, 1.0, 2.0, 3.0])
    cal = RankCalibrator().fit(train)

    out = cal.transform([0.0, 1.5, 3.0])
    assert out.shape == (3,)
    assert out.dtype == np.float64
    assert out.tolist() == pytest.approx([0.25, 0.5, 1.0])

