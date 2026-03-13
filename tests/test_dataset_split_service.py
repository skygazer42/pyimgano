from __future__ import annotations

import numpy as np

from pyimgano.services.dataset_split_service import LoadedBenchmarkSplit, load_benchmark_style_split


def test_load_benchmark_style_split_delegates_plain_loader(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_load_benchmark_split(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))

        class _Split:
            train_paths = ["train_a.png"]
            test_paths = ["test_a.png"]
            test_labels = np.asarray([0], dtype=np.int64)
            test_masks = None

        return _Split()

    import pyimgano.pipelines.mvtec_visa as mvtec_visa_module

    monkeypatch.setattr(mvtec_visa_module, "load_benchmark_split", _fake_load_benchmark_split)

    loaded = load_benchmark_style_split(
        dataset="custom",
        root="/tmp/custom",
        category="custom",
        resize=(16, 16),
        load_masks=True,
    )

    assert isinstance(loaded, LoadedBenchmarkSplit)
    assert loaded.pixel_skip_reason is None
    assert loaded.split.train_paths == ["train_a.png"]
    assert loaded.split.test_paths == ["test_a.png"]
    assert len(calls) == 1
    assert calls[0]["dataset"] == "custom"


def test_load_benchmark_style_split_delegates_manifest_loader(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_load_manifest_benchmark_split(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))

        class _ManifestSplit:
            train_paths = ["train_a.png"]
            calibration_paths = ["cal_a.png"]
            test_paths = ["test_a.png", "test_b.png"]
            test_labels = np.asarray([0, 1], dtype=np.int64)
            test_masks = np.ones((2, 4, 4), dtype=np.uint8)
            pixel_skip_reason = "manifest-pixel-skip"

        return _ManifestSplit()

    import pyimgano.datasets.manifest as manifest_module

    monkeypatch.setattr(
        manifest_module,
        "load_manifest_benchmark_split",
        _fake_load_manifest_benchmark_split,
    )

    loaded = load_benchmark_style_split(
        dataset="manifest",
        root="/tmp/root",
        manifest_path="/tmp/manifest.jsonl",
        category="bottle",
        resize=(16, 16),
        load_masks=True,
        seed=7,
        manifest_split_seed=None,
        manifest_test_normal_fraction=0.3,
    )

    assert isinstance(loaded, LoadedBenchmarkSplit)
    assert loaded.pixel_skip_reason == "manifest-pixel-skip"
    assert loaded.split.train_paths == ["train_a.png"]
    assert loaded.split.test_paths == ["test_a.png", "test_b.png"]
    assert np.asarray(loaded.split.test_labels).tolist() == [0, 1]
    assert len(calls) == 1
    assert calls[0]["manifest_path"] == "/tmp/manifest.jsonl"
    assert calls[0]["root_fallback"] == "/tmp/root"
    assert calls[0]["category"] == "bottle"
