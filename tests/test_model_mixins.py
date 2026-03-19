from __future__ import annotations

import math

import pytest


def test_ensure_int_validation() -> None:
    from pyimgano.models.mixins import ensure_int

    assert ensure_int(3, name="k", low=1) == 3
    try:
        ensure_int(0, name="k", low=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for low bound")


def test_ensure_contamination_validation() -> None:
    from pyimgano.models.mixins import ensure_contamination

    assert math.isclose(ensure_contamination(0.1), 0.1)
    try:
        ensure_contamination(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for contamination=0.0")
