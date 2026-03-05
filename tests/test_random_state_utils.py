from __future__ import annotations

import numpy as np


def test_check_random_state_int_is_deterministic() -> None:
    from pyimgano.utils.random_state import check_random_state

    rs1 = check_random_state(0)
    rs2 = check_random_state(0)

    a = rs1.normal(size=5)
    b = rs2.normal(size=5)
    assert np.allclose(a, b)


def test_check_random_state_pass_through() -> None:
    from pyimgano.utils.random_state import check_random_state

    rs = np.random.RandomState(123)
    assert check_random_state(rs) is rs


def test_check_random_state_none_returns_randomstate() -> None:
    from pyimgano.utils.random_state import check_random_state

    rs = check_random_state(None)
    assert isinstance(rs, np.random.RandomState)
