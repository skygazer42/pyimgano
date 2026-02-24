import pytest

from pyimgano.calibration.score_threshold import resolve_calibration_quantile


def test_resolve_calibration_quantile_prefers_explicit() -> None:
    class _Det:
        contamination = 0.1

    q, src = resolve_calibration_quantile(_Det(), calibration_quantile=0.5)
    assert q == pytest.approx(0.5)
    assert src == "explicit"


def test_resolve_calibration_quantile_uses_contamination() -> None:
    class _Det:
        contamination = 0.2

    q, src = resolve_calibration_quantile(_Det(), calibration_quantile=None)
    assert q == pytest.approx(0.8)
    assert src == "contamination"


def test_resolve_calibration_quantile_falls_back_when_contamination_missing() -> None:
    class _Det:
        pass

    q, src = resolve_calibration_quantile(_Det(), calibration_quantile=None, fallback=0.9)
    assert q == pytest.approx(0.9)
    assert src == "fallback"


def test_resolve_calibration_quantile_rejects_invalid_explicit() -> None:
    class _Det:
        contamination = 0.1

    with pytest.raises(ValueError, match="calibration_quantile must be in \\(0,1\\)"):
        resolve_calibration_quantile(_Det(), calibration_quantile=1.0)

