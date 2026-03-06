import numpy as np
import pytest

from pyimgano.inputs.torch_ops import to_torch_chw_float

pytest.importorskip("torch")


def test_to_torch_chw_float_shapes():
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    t = to_torch_chw_float(rgb, normalize=None)
    assert tuple(t.shape) == (3, 10, 20)
