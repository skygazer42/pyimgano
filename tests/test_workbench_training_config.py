import pytest

from pyimgano.workbench.config import WorkbenchConfig


def test_workbench_training_config_defaults():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
        }
    )

    assert cfg.training.enabled is False
    assert cfg.training.epochs is None
    assert cfg.training.lr is None
    assert cfg.training.checkpoint_name == "model.pt"


def test_workbench_training_config_parses_section_and_normalizes_types():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "training": {
                "enabled": True,
                "epochs": "2",
                "lr": "0.001",
                "checkpoint_name": "ae.pt",
            },
        }
    )

    assert cfg.training.enabled is True
    assert cfg.training.epochs == 2
    assert cfg.training.lr == 0.001
    assert cfg.training.checkpoint_name == "ae.pt"


def test_workbench_training_config_rejects_invalid_checkpoint_name():
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
        "training": {"enabled": True, "checkpoint_name": "a/b.pt"},
    }
    with pytest.raises(ValueError, match="checkpoint_name"):
        WorkbenchConfig.from_dict(raw)


def test_workbench_training_config_rejects_non_positive_values():
    base = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
    }

    with pytest.raises(ValueError, match="training\\.epochs"):
        WorkbenchConfig.from_dict({**base, "training": {"epochs": 0}})

    with pytest.raises(ValueError, match="training\\.lr"):
        WorkbenchConfig.from_dict({**base, "training": {"lr": 0}})

