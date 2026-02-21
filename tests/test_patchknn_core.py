import numpy as np
import pytest

from pyimgano.models.patchknn_core import aggregate_patch_scores, reshape_patch_scores


def test_aggregate_patch_scores_topk_mean():
    scores = np.arange(100, dtype=np.float32)
    out = aggregate_patch_scores(scores, method="topk_mean", topk=0.1)
    assert 90.0 <= out <= 99.0


def test_reshape_patch_scores_requires_exact_count():
    with pytest.raises(ValueError, match="Expected"):
        reshape_patch_scores(np.ones((3,), dtype=np.float32), grid_h=2, grid_w=2)

