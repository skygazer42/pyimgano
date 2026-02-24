import numpy as np

from pyimgano.defects.mask import anomaly_map_to_binary_mask


def test_anomaly_map_to_binary_mask_thresholds_to_uint8_255() -> None:
    m = np.asarray([[0.1, 0.9]], dtype=np.float32)
    mask = anomaly_map_to_binary_mask(m, pixel_threshold=0.5)
    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 255]]

