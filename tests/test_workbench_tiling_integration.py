import numpy as np

from pyimgano.workbench.adaptation import TilingConfig, apply_tiling


def test_apply_tiling_passthrough_when_disabled():
    detector = object()
    wrapped = apply_tiling(detector, TilingConfig(tile_size=None))
    assert wrapped is detector


def test_apply_tiling_wraps_with_tiled_detector_and_aggregates_scores():
    class _DummyDetector:
        def fit(self, X, y=None, **kwargs):  # noqa: ANN001, ANN003
            return self

        def decision_function(self, X):  # noqa: ANN001
            items = list(X)
            scores = [float(np.asarray(item).mean()) for item in items]
            return np.asarray(scores, dtype=np.float32)

    detector = _DummyDetector()
    wrapped = apply_tiling(
        detector,
        TilingConfig(
            tile_size=2,
            stride=2,
            score_reduce="max",
            score_topk=0.1,
            map_reduce="max",
        ),
    )

    img = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.float32,
    )
    scores = wrapped.decision_function([img])
    assert np.allclose(scores, np.asarray([3.0], dtype=np.float32))

