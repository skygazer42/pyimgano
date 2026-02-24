import numpy as np
import pytest

from pyimgano.defects.roi import clamp_roi_xyxy_norm, roi_mask_from_xyxy_norm


def test_clamp_roi_xyxy_norm_orders_and_clamps() -> None:
    roi = clamp_roi_xyxy_norm([1.2, -0.1, 0.7, 0.3])
    assert roi == pytest.approx([0.7, 0.0, 1.0, 0.3])


def test_roi_mask_from_xyxy_norm_shape_and_coverage() -> None:
    mask = roi_mask_from_xyxy_norm((10, 20), [0.25, 0.2, 0.75, 0.8])
    assert mask.shape == (10, 20)
    assert mask.dtype == np.uint8

    # ROI should include some pixels and exclude some pixels.
    assert int(mask.sum()) > 0
    assert int(mask.sum()) < int(mask.size)

