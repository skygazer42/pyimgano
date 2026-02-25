import pytest

from pyimgano.workbench.config import WorkbenchConfig


def test_workbench_config_from_dict_normalizes_types():
    raw = {
        "recipe": "industrial-adapt",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": "/tmp/data",
            "category": "all",
            "resize": [128, 256],
            "input_mode": "paths",
            "limit_train": 1,
            "limit_test": 2,
        },
        "model": {
            "name": "vision_patchcore",
            "device": "cpu",
            "preset": "industrial-fast",
            "pretrained": False,
            "contamination": 0.2,
            "model_kwargs": {"k": 1},
            "checkpoint_path": "/tmp/ckpt.pt",
        },
        "output": {
            "output_dir": "out",
            "save_run": False,
            "per_image_jsonl": False,
        },
        "defects": {
            "enabled": True,
            "pixel_threshold": 0.5,
            "pixel_threshold_strategy": "fixed",
            "pixel_normal_quantile": 0.999,
            "mask_format": "png",
            "roi_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            "min_area": 10,
            "min_score_max": 0.9,
            "min_score_mean": 0.5,
            "open_ksize": 3,
            "close_ksize": 5,
            "fill_holes": True,
            "max_regions": 7,
        },
    }

    cfg = WorkbenchConfig.from_dict(raw)
    assert cfg.recipe == "industrial-adapt"
    assert cfg.seed == 123
    assert cfg.dataset.name == "custom"
    assert cfg.dataset.resize == (128, 256)
    assert cfg.model.model_kwargs == {"k": 1}
    assert cfg.output.output_dir == "out"
    assert cfg.output.save_run is False
    assert cfg.output.per_image_jsonl is False
    assert cfg.defects.enabled is True
    assert cfg.defects.pixel_threshold == 0.5
    assert cfg.defects.pixel_threshold_strategy == "fixed"
    assert cfg.defects.pixel_normal_quantile == 0.999
    assert cfg.defects.mask_format == "png"
    assert cfg.defects.roi_xyxy_norm == pytest.approx((0.1, 0.2, 0.8, 0.9))
    assert cfg.defects.min_area == 10
    assert cfg.defects.min_score_max == pytest.approx(0.9)
    assert cfg.defects.min_score_mean == pytest.approx(0.5)
    assert cfg.defects.open_ksize == 3
    assert cfg.defects.close_ksize == 5
    assert cfg.defects.fill_holes is True
    assert cfg.defects.max_regions == 7


def test_workbench_config_defaults():
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
    }
    cfg = WorkbenchConfig.from_dict(raw)
    assert cfg.recipe == "industrial-adapt"
    assert cfg.seed is None
    assert cfg.dataset.category == "all"
    assert cfg.dataset.resize == (256, 256)
    assert cfg.output.save_run is True
    assert cfg.output.per_image_jsonl is True
    assert cfg.defects.enabled is False
    assert cfg.defects.pixel_threshold is None
    assert cfg.defects.roi_xyxy_norm is None


def test_workbench_config_invalid_resize_raises():
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data", "resize": [1, 2, 3]},
        "model": {"name": "vision_patchcore"},
    }
    with pytest.raises(ValueError):
        WorkbenchConfig.from_dict(raw)


def test_workbench_config_manifest_requires_manifest_path():
    raw = {
        "dataset": {"name": "manifest", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
    }
    with pytest.raises(ValueError, match=r"dataset\.manifest_path is required"):
        WorkbenchConfig.from_dict(raw)


def test_workbench_config_manifest_parses_split_policy_defaults():
    raw = {
        "seed": 7,
        "dataset": {
            "name": "manifest",
            "root": "/tmp/data",
            "manifest_path": "/tmp/manifest.jsonl",
            "split_policy": {"test_normal_fraction": 0.3},
        },
        "model": {"name": "vision_patchcore"},
    }
    cfg = WorkbenchConfig.from_dict(raw)
    assert cfg.seed == 7
    assert cfg.dataset.name == "manifest"
    assert cfg.dataset.manifest_path == "/tmp/manifest.jsonl"
    assert cfg.dataset.split_policy.seed == 7
    assert cfg.dataset.split_policy.mode == "benchmark"
    assert cfg.dataset.split_policy.scope == "category"
    assert cfg.dataset.split_policy.test_normal_fraction == 0.3
