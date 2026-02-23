import numpy as np


def _write_png(path, *, value: int = 128) -> None:
    from PIL import Image

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def test_feature_cache_reuses_disk_features(tmp_path) -> None:
    from pyimgano.models.baseml import BaseVisionDetector

    class _CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, paths):  # noqa: ANN001, ANN201 - test helper
            self.calls += 1
            feats: list[list[float]] = []
            for p in list(paths):
                s = str(p)
                feats.append([float(len(s)), float(sum(ord(ch) for ch in s) % 1000)])
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

    p0 = tmp_path / "img0.png"
    p1 = tmp_path / "img1.png"
    _write_png(p0, value=120)
    _write_png(p1, value=240)
    paths = [str(p0), str(p1)]

    cache_dir = tmp_path / "cache"

    extractor = _CountingExtractor()
    det = _DummyDetector(feature_extractor=extractor)
    det.set_feature_cache(cache_dir)

    det.fit(paths)
    assert extractor.calls == 1

    # Should reuse cached features (no additional extract calls).
    det.decision_function(paths)
    det.decision_function(paths)
    assert extractor.calls == 1

    # Cache persists across detector instances.
    extractor2 = _CountingExtractor()
    det2 = _DummyDetector(feature_extractor=extractor2)
    det2.set_feature_cache(cache_dir)
    det2.decision_function(paths)
    assert extractor2.calls == 0


def test_run_benchmark_cache_dir_writes_feature_files(tmp_path) -> None:
    import pyimgano.models  # noqa: F401 - registry population

    from pyimgano.pipelines.run_benchmark import run_benchmark

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    cache_dir = tmp_path / "cache"
    payload = run_benchmark(
        dataset="custom",
        root=str(root),
        category="custom",
        model="vision_ecod",
        device="cpu",
        pretrained=False,
        save_run=False,
        per_image_jsonl=False,
        limit_train=1,
        limit_test=2,
        cache_dir=str(cache_dir),
    )
    assert payload["dataset"] == "custom"
    assert list(cache_dir.rglob("*.npy"))
