import numpy as np

from pyimgano.pipelines.mvtec_visa import BenchmarkSplit, evaluate_split


class _DummyMapDetector:
    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        paths = list(X)
        # Deterministic scores: make later item more anomalous
        return np.linspace(0.0, 1.0, num=len(paths), dtype=np.float64)

    def get_anomaly_map(self, image_path: str):
        # Return a tiny heatmap to exercise resizing logic.
        if "anomaly" in image_path:
            return np.ones((2, 2), dtype=np.float32)
        return np.zeros((2, 2), dtype=np.float32)


class _DummyBatchMapDetector:
    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        paths = list(X)
        return np.linspace(0.0, 1.0, num=len(paths), dtype=np.float64)

    def predict_anomaly_map(self, X):
        paths = list(X)
        maps = []
        for path in paths:
            if "anomaly" in path:
                maps.append(np.ones((2, 2), dtype=np.float32))
            else:
                maps.append(np.zeros((2, 2), dtype=np.float32))
        return np.stack(maps, axis=0)


def _make_split() -> BenchmarkSplit:
    test_paths = ["test_normal_0.png", "test_anomaly_0.png"]
    test_labels = np.array([0, 1], dtype=int)
    test_masks = np.stack(
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
        ],
        axis=0,
    )
    return BenchmarkSplit(
        train_paths=["train_normal_0.png"],
        test_paths=test_paths,
        test_labels=test_labels,
        test_masks=test_masks,
    )


def test_evaluate_split_auto_pixel_scores_get_anomaly_map():
    split = _make_split()
    det = _DummyMapDetector()

    results = evaluate_split(det, split, compute_pixel_scores=True)
    assert "pixel_metrics" in results
    assert 0.0 <= results["pixel_metrics"]["pixel_auroc"] <= 1.0
    assert 0.0 <= results["pixel_metrics"]["pixel_average_precision"] <= 1.0
    assert 0.0 <= results["pixel_metrics"]["aupro"] <= 1.0


def test_evaluate_split_auto_pixel_scores_predict_anomaly_map():
    split = _make_split()
    det = _DummyBatchMapDetector()

    results = evaluate_split(det, split, compute_pixel_scores=True)
    assert "pixel_metrics" in results


def test_evaluate_split_no_pixel_scores_when_disabled():
    split = _make_split()
    det = _DummyMapDetector()

    results = evaluate_split(det, split, compute_pixel_scores=False)
    assert "pixel_metrics" not in results

