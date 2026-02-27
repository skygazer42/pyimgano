from __future__ import annotations

import numpy as np


def test_reduce_anomaly_map_topk_mean() -> None:
    from pyimgano.postprocess.reducers import reduce_anomaly_map

    m = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    # top 50% pixels are [2,3] => mean=2.5
    s = reduce_anomaly_map(m, method="topk_mean", topk=0.5)
    assert abs(float(s) - 2.5) < 1e-9


def test_reduce_anomaly_map_area() -> None:
    from pyimgano.postprocess.reducers import reduce_anomaly_map

    m = np.array([[0.0, 0.6], [0.4, 0.9]], dtype=np.float32)
    s = reduce_anomaly_map(m, method="area", area_threshold=0.5)
    assert abs(float(s) - 0.5) < 1e-9

