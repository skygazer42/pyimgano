from __future__ import annotations

import numpy as np


def test_feature_cache_reuses_numpy_image_features(tmp_path) -> None:
    from pyimgano.models.baseml import BaseVisionDetector

    class _CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, inputs):  # noqa: ANN001, ANN201 - test helper
            self.calls += 1
            feats: list[list[float]] = []
            for img in inputs:
                arr = np.asarray(img)
                feats.append([float(arr.mean()), float(arr.std())])
            return np.asarray(feats, dtype=np.float32)

    class _StubBackend:
        def fit(self, X):  # noqa: ANN001, ANN201 - test helper
            self.decision_scores_ = np.asarray(X).sum(axis=1)

        def decision_function(self, X):  # noqa: ANN001, ANN201 - test helper
            return np.asarray(X).sum(axis=1)

    class _DummyDetector(BaseVisionDetector):
        def __init__(self, *, contamination=0.1, feature_extractor=None):  # noqa: ANN001
            super().__init__(contamination=contamination, feature_extractor=feature_extractor)

        def _build_detector(self):  # noqa: ANN201
            return _StubBackend()

    imgs = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.ones((16, 16, 3), dtype=np.uint8) * 255,
    ]

    cache_dir = tmp_path / "cache"

    extractor = _CountingExtractor()
    det = _DummyDetector(feature_extractor=extractor)
    det.set_feature_cache(cache_dir)

    det.fit(imgs)
    assert extractor.calls == 1

    det.decision_function(imgs)
    det.decision_function(imgs)
    assert extractor.calls == 1

    # Cache persists across detector instances (same extractor fingerprint).
    extractor2 = _CountingExtractor()
    det2 = _DummyDetector(feature_extractor=extractor2)
    det2.set_feature_cache(cache_dir)
    det2.decision_function(imgs)
    assert extractor2.calls == 0
