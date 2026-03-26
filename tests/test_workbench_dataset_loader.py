from __future__ import annotations

from pathlib import Path

import numpy as np

from pyimgano.datasets.manifest import ManifestBenchmarkSplit
from pyimgano.pipelines.mvtec_visa import BenchmarkSplit
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.dataset_loader import list_workbench_categories, load_workbench_split


def test_load_workbench_split_manifest_paths_preserves_manifest_fields(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_load_manifest_benchmark_split(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))
        return ManifestBenchmarkSplit(
            train_paths=["train_a.png"],
            calibration_paths=["cal_a.png"],
            test_paths=["test_a.png", "test_b.png"],
            test_labels=np.asarray([0, 1], dtype=np.int64),
            test_masks=np.ones((2, 4, 4), dtype=np.uint8),
            pixel_skip_reason="manifest-skip-reason",
            test_meta=[{"part": "A"}, None],
        )

    import pyimgano.datasets.manifest as manifest_module

    monkeypatch.setattr(
        manifest_module,
        "load_manifest_benchmark_split",
        _fake_load_manifest_benchmark_split,
    )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 11,
            "dataset": {
                "name": "manifest",
                "root": str(tmp_path),
                "manifest_path": str(manifest_path),
                "category": "bottle",
                "resize": [32, 32],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    split = load_workbench_split(config=cfg, category="bottle", load_masks=True)

    assert split.train_inputs == ["train_a.png"]
    assert split.calibration_inputs == ["cal_a.png"]
    assert split.test_inputs == ["test_a.png", "test_b.png"]
    assert split.input_format is None
    assert split.pixel_skip_reason == "manifest-skip-reason"
    assert split.test_meta == [{"part": "A"}, None]
    assert len(calls) == 1
    assert calls[0]["manifest_path"] == str(manifest_path)
    assert calls[0]["category"] == "bottle"


def test_load_workbench_split_manifest_paths_uses_split_policy_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_load_manifest_benchmark_split(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))
        return ManifestBenchmarkSplit(
            train_paths=["train_a.png"],
            calibration_paths=["cal_a.png"],
            test_paths=["test_a.png"],
            test_labels=np.asarray([0], dtype=np.int64),
            test_masks=None,
            pixel_skip_reason=None,
            test_meta=None,
        )

    import pyimgano.datasets.manifest as manifest_module
    import pyimgano.workbench.dataset_loader as loader_module
    from pyimgano.datasets.manifest import ManifestSplitPolicy

    monkeypatch.setattr(
        manifest_module,
        "load_manifest_benchmark_split",
        _fake_load_manifest_benchmark_split,
    )

    def _fake_build_manifest_split_policy(*, config):  # noqa: ANN001 - test seam
        assert str(config.dataset.name) == "manifest"
        return ManifestSplitPolicy(
            mode="benchmark",
            scope="dataset",
            seed=333,
            test_normal_fraction=0.4,
        )

    monkeypatch.setattr(
        loader_module, "build_manifest_split_policy", _fake_build_manifest_split_policy
    )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 11,
            "dataset": {
                "name": "manifest",
                "root": str(tmp_path),
                "manifest_path": str(manifest_path),
                "category": "bottle",
                "resize": [32, 32],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    split = load_workbench_split(config=cfg, category="bottle", load_masks=False)

    assert split.train_inputs == ["train_a.png"]
    assert len(calls) == 1
    assert calls[0]["split_policy"].seed == 333
    assert calls[0]["split_policy"].scope == "dataset"


def test_load_workbench_split_paths_uses_benchmark_split_loader(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_load_benchmark_split(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))
        return BenchmarkSplit(
            train_paths=["train_a.png", "train_b.png"],
            test_paths=["test_a.png"],
            test_labels=np.asarray([1], dtype=np.int64),
            test_masks=None,
        )

    import pyimgano.pipelines.mvtec_visa as mvtec_visa_module

    monkeypatch.setattr(mvtec_visa_module, "load_benchmark_split", _fake_load_benchmark_split)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path),
                "category": "custom",
                "resize": [24, 24],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    split = load_workbench_split(config=cfg, category="custom", load_masks=True)

    assert split.train_inputs == ["train_a.png", "train_b.png"]
    assert split.calibration_inputs == ["train_a.png", "train_b.png"]
    assert split.test_inputs == ["test_a.png"]
    assert split.input_format is None
    assert split.pixel_skip_reason is None
    assert split.test_meta is None
    assert len(calls) == 1
    assert calls[0]["dataset"] == "custom"
    assert calls[0]["category"] == "custom"


def test_load_workbench_split_numpy_mode_normalizes_array_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    class _DummyDataset:
        def get_train_data(self):
            return np.asarray(
                [
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.ones((4, 4, 3), dtype=np.uint8),
                ]
            )

        def get_test_data(self):
            test = np.asarray(
                [
                    np.full((4, 4, 3), fill_value=2, dtype=np.uint8),
                    np.full((4, 4, 3), fill_value=3, dtype=np.uint8),
                ]
            )
            labels = np.asarray([0, 1], dtype=np.int64)
            masks = np.ones((2, 4, 4), dtype=np.uint8)
            return test, labels, masks

    calls: list[dict[str, object]] = []

    def _fake_load_dataset(*args, **kwargs):  # noqa: ANN002, ANN003 - test seam
        calls.append({"args": args, "kwargs": dict(kwargs)})
        return _DummyDataset()

    import pyimgano.datasets as datasets_module

    monkeypatch.setattr(datasets_module, "load_dataset", _fake_load_dataset)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "numpy",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    split = load_workbench_split(config=cfg, category="custom", load_masks=True)

    assert split.input_format == "rgb_u8_hwc"
    assert len(split.train_inputs) == 2
    assert len(split.calibration_inputs) == 2
    assert len(split.test_inputs) == 2
    assert isinstance(split.train_inputs[0], np.ndarray)
    assert isinstance(split.test_inputs[1], np.ndarray)
    assert np.asarray(split.test_labels).tolist() == [0, 1]
    assert split.test_masks is not None
    assert len(calls) == 1
    assert calls[0]["args"] == ("custom", str(tmp_path))
    assert calls[0]["kwargs"]["category"] == "custom"


def test_load_workbench_split_rejects_manifest_numpy_mode(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "manifest",
                "root": str(tmp_path),
                "manifest_path": str(manifest_path),
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "numpy",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    try:
        load_workbench_split(config=cfg, category="bottle", load_masks=True)
    except ValueError as exc:
        assert "dataset.name='manifest'" in str(exc)
    else:  # pragma: no cover - defensive assertion style
        raise AssertionError("Expected ValueError for manifest numpy mode")


def test_list_workbench_categories_delegates_manifest_listing(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_list_dataset_categories(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))
        return ["bottle", "cable"]

    import pyimgano.datasets.catalog as catalog_module

    monkeypatch.setattr(catalog_module, "list_dataset_categories", _fake_list_dataset_categories)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "manifest",
                "root": str(tmp_path / "dataset-root"),
                "manifest_path": str(manifest_path),
                "category": "all",
                "resize": [32, 32],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    categories = list_workbench_categories(config=cfg)

    assert categories == ["bottle", "cable"]
    assert calls == [
        {
            "dataset": "manifest",
            "root": str(tmp_path / "dataset-root"),
            "manifest_path": str(manifest_path),
        }
    ]


def test_list_workbench_categories_delegates_non_manifest_listing(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_list_dataset_categories(**kwargs):  # noqa: ANN003 - test seam
        calls.append(dict(kwargs))
        return ["custom"]

    import pyimgano.datasets.catalog as catalog_module

    monkeypatch.setattr(catalog_module, "list_dataset_categories", _fake_list_dataset_categories)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path),
                "category": "all",
                "resize": [24, 24],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    categories = list_workbench_categories(config=cfg)

    assert categories == ["custom"]
    assert calls == [{"dataset": "custom", "root": str(tmp_path), "manifest_path": None}]
