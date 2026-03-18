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
    assert cfg.training.validation_fraction is None
    assert cfg.training.early_stopping_patience is None
    assert cfg.training.early_stopping_min_delta is None
    assert cfg.training.max_steps is None
    assert cfg.training.max_train_samples is None
    assert cfg.training.batch_size is None
    assert cfg.training.num_workers is None
    assert cfg.training.weight_decay is None
    assert cfg.training.optimizer_name is None
    assert cfg.training.optimizer_momentum is None
    assert cfg.training.optimizer_nesterov is None
    assert cfg.training.optimizer_dampening is None
    assert cfg.training.adam_beta1 is None
    assert cfg.training.adam_beta2 is None
    assert cfg.training.adam_amsgrad is None
    assert cfg.training.optimizer_eps is None
    assert cfg.training.rmsprop_alpha is None
    assert cfg.training.rmsprop_centered is None
    assert cfg.training.scheduler_name is None
    assert cfg.training.scheduler_step_size is None
    assert cfg.training.scheduler_gamma is None
    assert cfg.training.scheduler_t_max is None
    assert cfg.training.scheduler_eta_min is None
    assert cfg.training.scheduler_patience is None
    assert cfg.training.scheduler_factor is None
    assert cfg.training.scheduler_min_lr is None
    assert cfg.training.scheduler_cooldown is None
    assert cfg.training.scheduler_threshold is None
    assert cfg.training.scheduler_threshold_mode is None
    assert cfg.training.scheduler_eps is None
    assert cfg.training.criterion_name is None
    assert cfg.training.shuffle_train is None
    assert cfg.training.drop_last is None
    assert cfg.training.pin_memory is None
    assert cfg.training.persistent_workers is None
    assert cfg.training.validation_split_seed is None
    assert cfg.training.warmup_epochs is None
    assert cfg.training.warmup_start_factor is None
    assert cfg.training.ema_enabled is None
    assert cfg.training.ema_decay is None
    assert cfg.training.ema_start_epoch is None
    assert cfg.training.scheduler_milestones is None
    assert cfg.training.resume_from_checkpoint is None
    assert cfg.training.checkpoint_name == "model.pt"
    assert cfg.training.tracker_backend is None
    assert cfg.training.tracker_dir is None
    assert cfg.training.tracker_project is None
    assert cfg.training.tracker_run_name is None
    assert cfg.training.tracker_mode is None
    assert cfg.training.callbacks == ()


def test_workbench_training_config_parses_section_and_normalizes_types():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "training": {
                "enabled": True,
                "epochs": "2",
                "lr": "0.001",
                "validation_fraction": "0.25",
                "early_stopping_patience": "3",
                "early_stopping_min_delta": "0.01",
                "max_steps": "12",
                "max_train_samples": "64",
                "batch_size": "8",
                "num_workers": "2",
                "weight_decay": "0.0005",
                "optimizer_name": "adamw",
                "optimizer_momentum": "0.85",
                "optimizer_nesterov": "true",
                "optimizer_dampening": "0.0",
                "adam_beta1": "0.81",
                "adam_beta2": "0.97",
                "adam_amsgrad": "true",
                "optimizer_eps": "0.000001",
                "rmsprop_alpha": "0.95",
                "rmsprop_centered": "false",
                "scheduler_name": "plateau",
                "scheduler_patience": "2",
                "scheduler_factor": "0.5",
                "scheduler_min_lr": "0.00001",
                "scheduler_cooldown": "1",
                "scheduler_threshold": "0.0005",
                "scheduler_threshold_mode": "abs",
                "scheduler_eps": "0.0000001",
                "scheduler_step_size": "2",
                "scheduler_gamma": "0.5",
                "criterion_name": "mae",
                "shuffle_train": "false",
                "drop_last": "true",
                "pin_memory": "true",
                "persistent_workers": "false",
                "validation_split_seed": "17",
                "warmup_epochs": "3",
                "warmup_start_factor": "0.25",
                "resume_from_checkpoint": "  /tmp/checkpoints/base.pt  ",
                "checkpoint_name": "ae.pt",
                "tracker_backend": "tensorboard",
                "tracker_dir": " ./runs/train ",
                "tracker_project": " pyimgano-dev ",
                "tracker_run_name": " run-a ",
                "tracker_mode": " offline ",
                "callbacks": ["metrics_logger"],
            },
        }
    )

    assert cfg.training.enabled is True
    assert cfg.training.epochs == 2
    assert cfg.training.lr == pytest.approx(0.001)
    assert cfg.training.validation_fraction == pytest.approx(0.25)
    assert cfg.training.early_stopping_patience == 3
    assert cfg.training.early_stopping_min_delta == pytest.approx(0.01)
    assert cfg.training.max_steps == 12
    assert cfg.training.max_train_samples == 64
    assert cfg.training.batch_size == 8
    assert cfg.training.num_workers == 2
    assert cfg.training.weight_decay == pytest.approx(0.0005)
    assert cfg.training.optimizer_name == "adamw"
    assert cfg.training.optimizer_momentum == pytest.approx(0.85)
    assert cfg.training.optimizer_nesterov is True
    assert cfg.training.optimizer_dampening == pytest.approx(0.0)
    assert cfg.training.adam_beta1 == pytest.approx(0.81)
    assert cfg.training.adam_beta2 == pytest.approx(0.97)
    assert cfg.training.adam_amsgrad is True
    assert cfg.training.optimizer_eps == pytest.approx(0.000001)
    assert cfg.training.rmsprop_alpha == pytest.approx(0.95)
    assert cfg.training.rmsprop_centered is False
    assert cfg.training.scheduler_name == "plateau"
    assert cfg.training.scheduler_step_size == 2
    assert cfg.training.scheduler_gamma == pytest.approx(0.5)
    assert cfg.training.scheduler_t_max is None
    assert cfg.training.scheduler_eta_min is None
    assert cfg.training.scheduler_patience == 2
    assert cfg.training.scheduler_factor == pytest.approx(0.5)
    assert cfg.training.scheduler_min_lr == pytest.approx(0.00001)
    assert cfg.training.scheduler_cooldown == 1
    assert cfg.training.scheduler_threshold == pytest.approx(0.0005)
    assert cfg.training.scheduler_threshold_mode == "abs"
    assert cfg.training.scheduler_eps == pytest.approx(0.0000001)
    assert cfg.training.criterion_name == "mae"
    assert cfg.training.shuffle_train is False
    assert cfg.training.drop_last is True
    assert cfg.training.pin_memory is True
    assert cfg.training.persistent_workers is False
    assert cfg.training.validation_split_seed == 17
    assert cfg.training.warmup_epochs == 3
    assert cfg.training.warmup_start_factor == pytest.approx(0.25)
    assert cfg.training.resume_from_checkpoint == "/tmp/checkpoints/base.pt"
    assert cfg.training.checkpoint_name == "ae.pt"
    assert cfg.training.tracker_backend == "tensorboard"
    assert cfg.training.tracker_dir == "./runs/train"
    assert cfg.training.tracker_project == "pyimgano-dev"
    assert cfg.training.tracker_run_name == "run-a"
    assert cfg.training.tracker_mode == "offline"
    assert cfg.training.callbacks == ("metrics_logger",)


def test_workbench_training_config_rejects_invalid_checkpoint_name():
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
        "training": {"enabled": True, "checkpoint_name": "a/b.pt"},
    }
    with pytest.raises(ValueError, match="checkpoint_name"):
        WorkbenchConfig.from_dict(raw)


def test_workbench_training_config_parses_multistep_scheduler_milestones():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "training": {
                "enabled": True,
                "scheduler_name": "multistep",
                "scheduler_milestones": ["2", "5"],
                "scheduler_gamma": "0.4",
            },
        }
    )

    assert cfg.training.enabled is True
    assert cfg.training.scheduler_name == "multistep"
    assert cfg.training.scheduler_milestones == (2, 5)
    assert cfg.training.scheduler_gamma == pytest.approx(0.4)


def test_workbench_training_config_parses_ema_strategy():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "training": {
                "enabled": True,
                "ema_enabled": "true",
                "ema_decay": "0.995",
                "ema_start_epoch": "2",
            },
        }
    )

    assert cfg.training.enabled is True
    assert cfg.training.ema_enabled is True
    assert cfg.training.ema_decay == pytest.approx(0.995)
    assert cfg.training.ema_start_epoch == 2


def test_workbench_training_config_parses_resource_profiler_callback():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "training": {
                "enabled": True,
                "callbacks": ["metrics_logger", "resource_profiler", "metrics_logger"],
            },
        }
    )

    assert cfg.training.callbacks == ("metrics_logger", "resource_profiler")


def test_workbench_training_config_parses_mlflow_tracker_backend():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "training": {
                "enabled": True,
                "tracker_backend": "mlflow",
                "tracker_project": "pyimgano-prod",
                "tracker_run_name": "run-42",
            },
        }
    )

    assert cfg.training.tracker_backend == "mlflow"
    assert cfg.training.tracker_project == "pyimgano-prod"
    assert cfg.training.tracker_run_name == "run-42"


def test_workbench_training_config_rejects_invalid_tracker_backend_and_callbacks():
    base = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
    }

    with pytest.raises(ValueError, match="training\\.tracker_backend"):
        WorkbenchConfig.from_dict(
            {**base, "training": {"enabled": True, "tracker_backend": "comet"}}
        )

    with pytest.raises(ValueError, match="training\\.callbacks"):
        WorkbenchConfig.from_dict(
            {**base, "training": {"enabled": True, "callbacks": ["metrics_logger", ""]}}
        )

    with pytest.raises(ValueError, match="training\\.callbacks"):
        WorkbenchConfig.from_dict(
            {**base, "training": {"enabled": True, "callbacks": ["resource_profiler", "unknown"]}}
        )


def test_workbench_training_config_rejects_non_positive_values():
    base = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
    }

    with pytest.raises(ValueError, match="training\\.epochs"):
        WorkbenchConfig.from_dict({**base, "training": {"epochs": 0}})

    with pytest.raises(ValueError, match="training\\.lr"):
        WorkbenchConfig.from_dict({**base, "training": {"lr": 0}})

    with pytest.raises(ValueError, match="training\\.validation_fraction"):
        WorkbenchConfig.from_dict({**base, "training": {"validation_fraction": 1.0}})

    with pytest.raises(ValueError, match="training\\.early_stopping_patience"):
        WorkbenchConfig.from_dict({**base, "training": {"early_stopping_patience": 0}})

    with pytest.raises(ValueError, match="training\\.early_stopping_min_delta"):
        WorkbenchConfig.from_dict({**base, "training": {"early_stopping_min_delta": -0.1}})

    with pytest.raises(ValueError, match="training\\.max_steps"):
        WorkbenchConfig.from_dict({**base, "training": {"max_steps": 0}})

    with pytest.raises(ValueError, match="training\\.max_train_samples"):
        WorkbenchConfig.from_dict({**base, "training": {"max_train_samples": 0}})

    with pytest.raises(ValueError, match="training\\.batch_size"):
        WorkbenchConfig.from_dict({**base, "training": {"batch_size": 0}})

    with pytest.raises(ValueError, match="training\\.num_workers"):
        WorkbenchConfig.from_dict({**base, "training": {"num_workers": -1}})

    with pytest.raises(ValueError, match="training\\.weight_decay"):
        WorkbenchConfig.from_dict({**base, "training": {"weight_decay": -0.01}})

    with pytest.raises(ValueError, match="training\\.optimizer_name"):
        WorkbenchConfig.from_dict({**base, "training": {"optimizer_name": "bogus"}})

    with pytest.raises(ValueError, match="training\\.optimizer_momentum"):
        WorkbenchConfig.from_dict({**base, "training": {"optimizer_momentum": -0.1}})

    with pytest.raises(ValueError, match="training\\.optimizer_nesterov"):
        WorkbenchConfig.from_dict({**base, "training": {"optimizer_nesterov": "maybe"}})

    with pytest.raises(ValueError, match="training\\.optimizer_dampening"):
        WorkbenchConfig.from_dict({**base, "training": {"optimizer_dampening": -0.1}})

    with pytest.raises(ValueError, match="training\\.adam_beta1"):
        WorkbenchConfig.from_dict({**base, "training": {"adam_beta1": 1.0}})

    with pytest.raises(ValueError, match="training\\.adam_beta2"):
        WorkbenchConfig.from_dict({**base, "training": {"adam_beta2": -0.1}})

    with pytest.raises(ValueError, match="training\\.optimizer_eps"):
        WorkbenchConfig.from_dict({**base, "training": {"optimizer_eps": 0}})

    with pytest.raises(ValueError, match="training\\.adam_amsgrad"):
        WorkbenchConfig.from_dict({**base, "training": {"adam_amsgrad": "maybe"}})

    with pytest.raises(ValueError, match="training\\.rmsprop_alpha"):
        WorkbenchConfig.from_dict({**base, "training": {"rmsprop_alpha": 1.0}})

    with pytest.raises(ValueError, match="training\\.rmsprop_centered"):
        WorkbenchConfig.from_dict({**base, "training": {"rmsprop_centered": "maybe"}})

    with pytest.raises(ValueError, match="optimizer_dampening"):
        WorkbenchConfig.from_dict(
            {
                **base,
                "training": {
                    "optimizer_name": "sgd",
                    "optimizer_nesterov": True,
                    "optimizer_dampening": 0.2,
                },
            }
        )

    with pytest.raises(ValueError, match="training\\.scheduler_name"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_name": "bogus"}})

    with pytest.raises(ValueError, match="training\\.scheduler_milestones"):
        WorkbenchConfig.from_dict(
            {
                **base,
                "training": {
                    "scheduler_name": "multistep",
                    "scheduler_milestones": [0, 2],
                },
            }
        )

    with pytest.raises(ValueError, match="training\\.scheduler_milestones"):
        WorkbenchConfig.from_dict(
            {
                **base,
                "training": {
                    "scheduler_name": "multistep",
                    "scheduler_milestones": [],
                },
            }
        )

    with pytest.raises(ValueError, match="training\\.scheduler_milestones"):
        WorkbenchConfig.from_dict(
            {
                **base,
                "training": {
                    "scheduler_name": "multistep",
                },
            }
        )

    with pytest.raises(ValueError, match="training\\.criterion_name"):
        WorkbenchConfig.from_dict({**base, "training": {"criterion_name": "bogus"}})

    with pytest.raises(ValueError, match="training\\.scheduler_step_size"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_step_size": 0}})

    with pytest.raises(ValueError, match="training\\.scheduler_gamma"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_gamma": 0}})

    with pytest.raises(ValueError, match="training\\.scheduler_t_max"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_t_max": 0}})

    with pytest.raises(ValueError, match="training\\.scheduler_eta_min"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_eta_min": -0.1}})

    with pytest.raises(ValueError, match="training\\.scheduler_patience"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_patience": -1}})

    with pytest.raises(ValueError, match="training\\.scheduler_factor"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_factor": 0}})

    with pytest.raises(ValueError, match="training\\.scheduler_min_lr"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_min_lr": -0.1}})

    with pytest.raises(ValueError, match="training\\.scheduler_cooldown"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_cooldown": -1}})

    with pytest.raises(ValueError, match="training\\.scheduler_threshold"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_threshold": -0.1}})

    with pytest.raises(ValueError, match="training\\.scheduler_threshold_mode"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_threshold_mode": "weird"}})

    with pytest.raises(ValueError, match="training\\.scheduler_eps"):
        WorkbenchConfig.from_dict({**base, "training": {"scheduler_eps": -1e-8}})

    with pytest.raises(ValueError, match="training\\.resume_from_checkpoint"):
        WorkbenchConfig.from_dict({**base, "training": {"resume_from_checkpoint": "   "}})

    with pytest.raises(ValueError, match="training\\.shuffle_train"):
        WorkbenchConfig.from_dict({**base, "training": {"shuffle_train": "maybe"}})

    with pytest.raises(ValueError, match="training\\.drop_last"):
        WorkbenchConfig.from_dict({**base, "training": {"drop_last": "maybe"}})

    with pytest.raises(ValueError, match="training\\.pin_memory"):
        WorkbenchConfig.from_dict({**base, "training": {"pin_memory": "maybe"}})

    with pytest.raises(ValueError, match="training\\.persistent_workers"):
        WorkbenchConfig.from_dict({**base, "training": {"persistent_workers": "maybe"}})

    with pytest.raises(ValueError, match="training\\.validation_split_seed"):
        WorkbenchConfig.from_dict({**base, "training": {"validation_split_seed": -1}})

    with pytest.raises(ValueError, match="training\\.warmup_epochs"):
        WorkbenchConfig.from_dict({**base, "training": {"warmup_epochs": 0}})

    with pytest.raises(ValueError, match="training\\.warmup_start_factor"):
        WorkbenchConfig.from_dict({**base, "training": {"warmup_start_factor": 1.1}})

    with pytest.raises(ValueError, match="training\\.ema_enabled"):
        WorkbenchConfig.from_dict({**base, "training": {"ema_enabled": "maybe"}})

    with pytest.raises(ValueError, match="training\\.ema_decay"):
        WorkbenchConfig.from_dict({**base, "training": {"ema_decay": 1.0}})

    with pytest.raises(ValueError, match="training\\.ema_start_epoch"):
        WorkbenchConfig.from_dict({**base, "training": {"ema_start_epoch": 0}})
