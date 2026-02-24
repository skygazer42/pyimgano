import numpy as np

from pyimgano.defects.map_ops import apply_roi_to_map, compute_roi_stats


def test_apply_roi_to_map_zeros_outside_roi() -> None:
    m = np.ones((4, 4), dtype=np.float32)
    out = apply_roi_to_map(m, roi_xyxy_norm=[0.5, 0.0, 1.0, 1.0])
    assert out.shape == (4, 4)
    assert float(out[:, :2].max()) == 0.0
    assert float(out[:, 2:].min()) == 1.0


def test_compute_roi_stats_returns_max_and_mean() -> None:
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    stats = compute_roi_stats(m, roi_xyxy_norm=[0.0, 0.0, 0.5, 1.0])
    assert set(stats) == {"max", "mean"}
    assert stats["max"] >= stats["mean"]

