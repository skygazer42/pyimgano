import numpy as np
import pytest

from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image, parse_image_format


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("bgr_u8_hwc", ImageFormat.BGR_U8_HWC),
        ("rgb_u8_hwc", ImageFormat.RGB_U8_HWC),
        ("rgb_f32_chw", ImageFormat.RGB_F32_CHW),
    ],
)
def test_parse_image_format(raw, expected):
    assert parse_image_format(raw) is expected


def test_parse_image_format_rejects_unknown():
    with pytest.raises(ValueError):
        parse_image_format("auto")


def test_normalize_bgr_u8_hwc_to_rgb_u8_hwc_swaps_channels():
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R
    rgb = normalize_numpy_image(bgr, input_format=ImageFormat.BGR_U8_HWC)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (2, 3, 3)
    assert np.all(rgb[..., 0] == 30)
    assert np.all(rgb[..., 1] == 20)
    assert np.all(rgb[..., 2] == 10)


def test_normalize_rgb_f32_chw_to_rgb_u8_hwc_scales_and_transposes():
    chw = np.ones((3, 4, 5), dtype=np.float32) * 0.5
    out = normalize_numpy_image(chw, input_format=ImageFormat.RGB_F32_CHW)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 128


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 1), (3, 3, 4)])
def test_normalize_rejects_bad_shapes(shape):
    arr = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        normalize_numpy_image(arr, input_format=ImageFormat.RGB_U8_HWC)
