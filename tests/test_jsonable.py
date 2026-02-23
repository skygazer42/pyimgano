from __future__ import annotations

from pathlib import Path

import numpy as np

from pyimgano.utils.jsonable import to_jsonable


def test_to_jsonable_converts_common_types(tmp_path) -> None:
    value = {
        "path": tmp_path / "x.bin",
        "arr": np.asarray([1, 2, 3], dtype=np.int32),
        "scalar": np.float64(0.5),
        "nested": {2: np.bool_(True), "t": (np.int64(1), np.int64(2))},
    }

    converted = to_jsonable(value)
    assert isinstance(converted, dict)

    assert converted["path"].endswith("x.bin")
    assert converted["arr"] == [1, 2, 3]
    assert converted["scalar"] == 0.5

    nested = converted["nested"]
    assert nested["2"] is True
    assert nested["t"] == [1, 2]


def test_to_jsonable_leaves_builtin_types_unchanged() -> None:
    assert to_jsonable(1) == 1
    assert to_jsonable(1.5) == 1.5
    assert to_jsonable("x") == "x"
    assert to_jsonable(None) is None
    assert to_jsonable(Path("a/b")) == "a/b"

