import numpy as np


def test_anomalydino_aggregate_topk_mean():
    from pyimgano.models.anomalydino import _aggregate_patch_scores

    patch_scores = np.arange(10, dtype=np.float32)
    assert _aggregate_patch_scores(patch_scores, method="topk_mean", topk=0.2) == 8.5
    assert _aggregate_patch_scores(patch_scores, method="topk_mean", topk=0.01) == 9.0


def test_anomalydino_reshape_patch_scores():
    from pyimgano.models.anomalydino import _reshape_patch_scores

    patch_scores = np.arange(6, dtype=np.float32)
    grid = _reshape_patch_scores(patch_scores, grid_h=2, grid_w=3)
    assert grid.shape == (2, 3)
    assert float(grid[0, 0]) == 0.0
    assert float(grid[1, 2]) == 5.0
