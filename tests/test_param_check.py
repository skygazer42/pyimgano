import pytest


def test_check_parameter_enforces_bounds() -> None:
    from pyimgano.utils.param_check import check_parameter

    # Inclusive lower bound
    check_parameter(1, low=1, param_name="x", include_left=True)
    with pytest.raises(ValueError, match="x"):
        check_parameter(0, low=1, param_name="x", include_left=True)

    # Exclusive lower bound
    with pytest.raises(ValueError, match="x"):
        check_parameter(1, low=1, param_name="x", include_left=False)

    # Inclusive upper bound
    check_parameter(1, high=1, param_name="x", include_right=True)
    with pytest.raises(ValueError, match="x"):
        check_parameter(2, high=1, param_name="x", include_right=True)

    # Exclusive upper bound
    with pytest.raises(ValueError, match="x"):
        check_parameter(1, high=1, param_name="x", include_right=False)


def test_core_imdd_rejects_invalid_n_iter() -> None:
    pytest.importorskip("numba")

    from pyimgano.models.imdd import CoreIMDD

    with pytest.raises(ValueError):
        CoreIMDD(n_iter=0)


def test_check_parameter_rejects_non_numeric_bounds() -> None:
    from pyimgano.utils.param_check import check_parameter

    with pytest.raises(TypeError, match="low must be a number"):
        check_parameter(1, low="a", param_name="x")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="high must be a number"):
        check_parameter(1, high="b", param_name="x")  # type: ignore[arg-type]


def test_check_parameter_rejects_inverted_bounds() -> None:
    from pyimgano.utils.param_check import check_parameter

    with pytest.raises(ValueError, match="low=3 > high=1"):
        check_parameter(2, low=3, high=1, param_name="x")
