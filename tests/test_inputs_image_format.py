import numpy as np
import pytest

from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image, parse_image_format


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("bgr_u8_hwc", ImageFormat.BGR_U8_HWC),
        ("gray_u8_hw", ImageFormat.GRAY_U8_HW),
        ("gray_u8_hwc1", ImageFormat.GRAY_U8_HWC1),
        ("rgb_u8_hwc", ImageFormat.RGB_U8_HWC),
        ("bgr_f32_hwc", ImageFormat.BGR_F32_HWC),
        ("rgb_f32_hwc", ImageFormat.RGB_F32_HWC),
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


def test_normalize_gray_u8_hw_to_rgb_u8_hwc_repeats_channels():
    gray = np.zeros((2, 3), dtype=np.uint8)
    gray[0, 0] = 10
    out = normalize_numpy_image(gray, input_format=ImageFormat.GRAY_U8_HW)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 10
    assert int(out[0, 0, 1]) == 10
    assert int(out[0, 0, 2]) == 10


def test_normalize_gray_u8_hwc1_to_rgb_u8_hwc_repeats_channels():
    gray = np.zeros((2, 3, 1), dtype=np.uint8)
    gray[0, 0, 0] = 42
    out = normalize_numpy_image(gray, input_format=ImageFormat.GRAY_U8_HWC1)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 42
    assert int(out[0, 0, 1]) == 42
    assert int(out[0, 0, 2]) == 42


def test_normalize_bgr_f32_hwc_to_rgb_u8_hwc_swaps_and_scales():
    bgr = np.zeros((2, 3, 3), dtype=np.float32)
    bgr[..., 0] = 0.1  # B
    bgr[..., 1] = 0.2  # G
    bgr[..., 2] = 0.3  # R
    out = normalize_numpy_image(bgr, input_format=ImageFormat.BGR_F32_HWC)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    # np.rint uses bankers rounding: 0.3*255=76.5 -> 76 (nearest even).
    assert int(out[0, 0, 0]) == 76  # R
    assert int(out[0, 0, 1]) == 51  # G
    assert int(out[0, 0, 2]) == 26  # B


def test_normalize_rgb_f32_hwc_to_rgb_u8_hwc_scales():
    rgb = np.ones((2, 3, 3), dtype=np.float32) * 0.5
    out = normalize_numpy_image(rgb, input_format=ImageFormat.RGB_F32_HWC)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 128


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 1), (3, 3, 4)])
def test_normalize_rejects_bad_shapes(shape):
    arr = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        normalize_numpy_image(arr, input_format=ImageFormat.RGB_U8_HWC)
