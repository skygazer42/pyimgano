from __future__ import annotations

import numpy as np


def test_array_feature_cache_reuses_disk_features(tmp_path) -> None:
    from pyimgano.cache.array_features import ArrayFeatureCache, CachedArrayFeatureExtractor
    from pyimgano.cache.features import fingerprint_feature_extractor

    class _CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, arrays):  # noqa: ANN001, ANN201 - test helper
            self.calls += 1
            feats: list[list[float]] = []
            for a in list(arrays):
                arr = np.asarray(a)
                feats.append([float(arr.shape[0]), float(arr.shape[1]), float(np.mean(arr))])
            return np.asarray(feats, dtype=np.float32)

    arrays = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.ones((16, 16, 3), dtype=np.uint8) * 255,
    ]

    base = _CountingExtractor()
    cache = ArrayFeatureCache(
        cache_dir=tmp_path / "cache",
        extractor_fingerprint=fingerprint_feature_extractor(base),
    )
    wrapped = CachedArrayFeatureExtractor(base_extractor=base, cache=cache)

    f0 = wrapped.extract(arrays)
    assert base.calls == 1
    assert f0.shape[0] == len(arrays)

    f1 = wrapped.extract(arrays)
    assert base.calls == 1
    assert np.allclose(f0, f1)

    # Cache persists across extractor instances (same fingerprint).
    base2 = _CountingExtractor()
    cache2 = ArrayFeatureCache(
        cache_dir=tmp_path / "cache",
        extractor_fingerprint=fingerprint_feature_extractor(base2),
    )
    wrapped2 = CachedArrayFeatureExtractor(base_extractor=base2, cache=cache2)
    _ = wrapped2.extract(arrays)
    assert base2.calls == 0

