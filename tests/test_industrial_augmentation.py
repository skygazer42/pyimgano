from __future__ import annotations

import numpy as np

from pyimgano.preprocessing.augmentation import jpeg_compress, random_channel_gain, vignette
from pyimgano.preprocessing.augmentation_pipeline import get_industrial_camera_robust_augmentation


def test_jpeg_compress_preserves_shape_and_dtype() -> None:
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out = jpeg_compress(img, quality=60)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_vignette_runs() -> None:
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out = vignette(img, strength=0.3)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_random_channel_gain_runs() -> None:
    img = np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
    out = random_channel_gain(img, gain_range=(0.9, 1.1))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_industrial_camera_robust_preset_is_callable() -> None:
    aug = get_industrial_camera_robust_augmentation()
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out = aug(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

