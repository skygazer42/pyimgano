import numpy as np
import pytest


def test_pairwise_distances_no_broadcast_rowwise_euclidean() -> None:
    from pyimgano.utils.pairwise import pairwise_distances_no_broadcast

    X = np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=float)
    Y = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)

    d = pairwise_distances_no_broadcast(X, Y)
    assert d.shape == (2,)
    assert np.allclose(d, np.asarray([0.0, 5.0]))


def test_pairwise_distances_no_broadcast_requires_matching_shapes() -> None:
    from pyimgano.utils.pairwise import pairwise_distances_no_broadcast

    X = np.zeros((2, 3), dtype=float)
    Y = np.zeros((2, 4), dtype=float)
    with pytest.raises(ValueError, match="same shape"):
        pairwise_distances_no_broadcast(X, Y)

