from __future__ import annotations

import numpy as np

from pyimgano.robustness.benchmark import run_robustness_benchmark


class _DummyMapDetector:
    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        items = list(X)
        return np.array([float(np.max(np.asarray(x))) for x in items], dtype=np.float64)

    def predict_anomaly_map(self, X):
        items = list(X)
        maps = [np.asarray(x)[..., 0].astype(np.float32) for x in items]
        return np.stack(maps, axis=0)


class _IdentityCorruption:
    name = "identity"

    def __call__(self, image, mask, *, severity: int, rng: np.random.Generator):
        _ = severity, rng
        return image, mask


def test_run_robustness_benchmark_schema_and_metrics() -> None:
    train_images = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]

    normal = np.zeros((16, 16, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[4:12, 4:12] = 255
    test_images = [normal, anomaly]

    test_labels = np.array([0, 1], dtype=int)
    test_masks = np.stack(
        [
            np.zeros((16, 16), dtype=np.uint8),
            np.pad(np.ones((8, 8), dtype=np.uint8), pad_width=4),
        ],
        axis=0,
    )

    det = _DummyMapDetector()
    report = run_robustness_benchmark(
        det,
        train_images=train_images,
        test_images=test_images,
        test_labels=test_labels,
        test_masks=test_masks,
        corruptions=[_IdentityCorruption()],
        severities=[1, 3],
        seed=0,
        pixel_segf1=True,
        pixel_threshold_strategy="normal_pixel_quantile",
        pixel_normal_quantile=0.999,
        calibration_fraction=0.5,
        calibration_seed=0,
    )

    assert "clean" in report
    assert "corruptions" in report
    assert "identity" in report["corruptions"]

    clean_metrics = report["clean"]["results"]["pixel_metrics"]
    assert clean_metrics["pixel_segf1"] == 1.0
    assert clean_metrics["bg_fpr"] == 0.0

    sev1 = report["corruptions"]["identity"]["severity_1"]["results"]["pixel_metrics"]
    sev3 = report["corruptions"]["identity"]["severity_3"]["results"]["pixel_metrics"]
    assert sev1["pixel_segf1"] == 1.0
    assert sev3["pixel_segf1"] == 1.0
