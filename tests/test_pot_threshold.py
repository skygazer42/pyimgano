import numpy as np


def test_fit_pot_threshold_returns_finite_threshold_and_info() -> None:
    from pyimgano.calibration.pot_threshold import fit_pot_threshold

    rng = np.random.RandomState(0)
    scores = rng.exponential(scale=1.0, size=500)

    thr, info = fit_pot_threshold(scores, alpha=0.1, tail_fraction=0.2, min_exceedances=20)
    assert np.isfinite(thr)
    assert isinstance(info, dict)
    assert "method" in info


def test_fit_pot_threshold_falls_back_when_tail_too_small() -> None:
    from pyimgano.calibration.pot_threshold import fit_pot_threshold

    scores = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    thr, info = fit_pot_threshold(scores, alpha=0.2, tail_fraction=0.2, min_exceedances=1000)
    assert info["method"] == "quantile_fallback"
    # Fallback quantile for alpha=0.2 => 80th percentile => 3.2
    assert thr == np.quantile(scores, 0.8)

