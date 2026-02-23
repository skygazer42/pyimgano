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


def test_workbench_config_invalid_resize_raises():
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data", "resize": [1, 2, 3]},
        "model": {"name": "vision_patchcore"},
    }
    with pytest.raises(ValueError):
        WorkbenchConfig.from_dict(raw)

