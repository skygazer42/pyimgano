from __future__ import annotations

import numpy as np


def test_stable_hash_str_is_deterministic() -> None:
    from pyimgano.utils.hash_utils import stable_hash_str

    h1 = stable_hash_str("hello")
    h2 = stable_hash_str("hello")
    assert h1 == h2
    assert h1 != stable_hash_str("hello!")


def test_stable_hash_json_is_deterministic_and_order_independent() -> None:
    from pyimgano.utils.hash_utils import stable_hash_json

    a = {"x": 1, "y": [2, 3], "z": {"k": "v"}}
    b = {"z": {"k": "v"}, "y": [2, 3], "x": 1}
    assert stable_hash_json(a) == stable_hash_json(b)
    assert stable_hash_json(a) != stable_hash_json({"x": 2})


def test_stable_hash_array_matches_on_equal_content() -> None:
    from pyimgano.utils.hash_utils import stable_hash_array

    x1 = np.arange(12, dtype=np.float32).reshape(3, 4)
    x2 = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert stable_hash_array(x1) == stable_hash_array(x2)


def test_stable_hash_array_changes_when_content_changes() -> None:
    from pyimgano.utils.hash_utils import stable_hash_array

    x1 = np.zeros((8, 8, 3), dtype=np.uint8)
    x2 = x1.copy()
    x2[0, 0, 0] = 1
    assert stable_hash_array(x1) != stable_hash_array(x2)


def test_stable_hash_array_handles_non_contiguous_views() -> None:
    from pyimgano.utils.hash_utils import stable_hash_array

    x = np.arange(100, dtype=np.int64).reshape(10, 10)
    view = x[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert stable_hash_array(view) == stable_hash_array(np.ascontiguousarray(view))

