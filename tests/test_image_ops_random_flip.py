from __future__ import annotations

import numpy as np
from PIL import Image


def _make_test_image() -> Image.Image:
    # Make a non-symmetric image so we can verify horizontal flips.
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[:, :2] = 10
    arr[:, 2:] = 200
    return Image.fromarray(arr, mode="RGB")


def test_random_horizontal_flip_prob_one_always_flips() -> None:
    from pyimgano.utils.image_ops import random_horizontal_flip

    img = _make_test_image()
    out = random_horizontal_flip(img, prob=1.0, rng=np.random.default_rng(0))
    expected = img.transpose(Image.FLIP_LEFT_RIGHT)
    assert np.array_equal(np.asarray(out), np.asarray(expected))


def test_random_horizontal_flip_prob_zero_never_flips() -> None:
    from pyimgano.utils.image_ops import random_horizontal_flip

    img = _make_test_image()
    out = random_horizontal_flip(img, prob=0.0, rng=np.random.default_rng(0))
    assert np.array_equal(np.asarray(out), np.asarray(img))
