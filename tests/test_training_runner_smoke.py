from __future__ import annotations

import pytest

from pyimgano.training.runner import micro_finetune


def test_micro_finetune_passes_supported_fit_kwargs():
    class _Detector:
        def __init__(self):
            self.called = False
            self.received = None

        def fit(
            self,
            X,
            *,
            epochs=None,
            lr=None,
            batch_size=None,
            num_workers=None,
            weight_decay=None,
            optimizer_name=None,
            optimizer_momentum=None,
            optimizer_nesterov=None,
            optimizer_dampening=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_amsgrad=None,
            optimizer_eps=None,
            rmsprop_alpha=None,
            rmsprop_centered=None,
            scheduler_name=None,
            scheduler_step_size=None,
            scheduler_gamma=None,
            scheduler_patience=None,
            scheduler_factor=None,
            scheduler_min_lr=None,
            scheduler_cooldown=None,
            scheduler_threshold=None,
            scheduler_threshold_mode=None,
            scheduler_eps=None,
            criterion_name=None,
            shuffle_train=None,
            drop_last=None,
            pin_memory=None,
            persistent_workers=None,
            warmup_epochs=None,
            warmup_start_factor=None,
            validation_inputs=None,
            max_steps=None,
            early_stopping_patience=None,
            early_stopping_min_delta=None,
        ):  # noqa: ANN001 - test stub
            self.called = True
            self.received = {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "weight_decay": weight_decay,
                "optimizer_name": optimizer_name,
                "optimizer_momentum": optimizer_momentum,
                "optimizer_nesterov": optimizer_nesterov,
                "optimizer_dampening": optimizer_dampening,
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_amsgrad": adam_amsgrad,
                "optimizer_eps": optimizer_eps,
                "rmsprop_alpha": rmsprop_alpha,
                "rmsprop_centered": rmsprop_centered,
                "scheduler_name": scheduler_name,
                "scheduler_step_size": scheduler_step_size,
                "scheduler_gamma": scheduler_gamma,
                "scheduler_patience": scheduler_patience,
                "scheduler_factor": scheduler_factor,
                "scheduler_min_lr": scheduler_min_lr,
                "scheduler_cooldown": scheduler_cooldown,
                "scheduler_threshold": scheduler_threshold,
                "scheduler_threshold_mode": scheduler_threshold_mode,
                "scheduler_eps": scheduler_eps,
                "criterion_name": criterion_name,
                "shuffle_train": shuffle_train,
                "drop_last": drop_last,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "warmup_epochs": warmup_epochs,
                "warmup_start_factor": warmup_start_factor,
                "n": len(list(X)),
                "validation_n": len(list(validation_inputs or [])),
                "max_steps": max_steps,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
            }
            return self

    det = _Detector()
    out = micro_finetune(
        det,
        ["a", "b", "c", "d"],
        seed=123,
        fit_kwargs={
            "epochs": 2,
            "lr": 1e-3,
            "batch_size": 8,
                "num_workers": 2,
                "weight_decay": 0.001,
                "optimizer_name": "adamw",
                "optimizer_momentum": 0.8,
                "optimizer_nesterov": True,
                "optimizer_dampening": 0.0,
                "adam_beta1": 0.82,
                "adam_beta2": 0.96,
                "adam_amsgrad": True,
                "optimizer_eps": 1e-6,
                "rmsprop_alpha": 0.95,
                "rmsprop_centered": False,
                "scheduler_name": "plateau",
                "scheduler_patience": 2,
                "scheduler_factor": 0.5,
                "scheduler_min_lr": 1e-5,
                "scheduler_cooldown": 1,
                "scheduler_threshold": 5e-4,
                "scheduler_threshold_mode": "abs",
                "scheduler_eps": 1e-7,
                "scheduler_step_size": 2,
                "scheduler_gamma": 0.5,
                "criterion_name": "mae",
                "shuffle_train": False,
                "drop_last": True,
                "pin_memory": True,
                "persistent_workers": True,
                "validation_fraction": 0.25,
                "warmup_epochs": 3,
                "warmup_start_factor": 0.25,
                "max_steps": 5,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.01,
            },
    )

    assert det.called is True
    assert det.received == {
        "epochs": 2,
        "lr": 1e-3,
        "batch_size": 8,
        "num_workers": 2,
        "weight_decay": 0.001,
        "optimizer_name": "adamw",
        "optimizer_momentum": 0.8,
        "optimizer_nesterov": True,
        "optimizer_dampening": 0.0,
        "adam_beta1": 0.82,
        "adam_beta2": 0.96,
        "adam_amsgrad": True,
        "optimizer_eps": 1e-6,
        "rmsprop_alpha": 0.95,
        "rmsprop_centered": False,
        "scheduler_name": "plateau",
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-5,
        "scheduler_cooldown": 1,
        "scheduler_threshold": 5e-4,
        "scheduler_threshold_mode": "abs",
        "scheduler_eps": 1e-7,
        "criterion_name": "mae",
        "shuffle_train": False,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.25,
        "n": 3,
        "validation_n": 1,
        "max_steps": 5,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.01,
    }
    assert out["dataset"] == {
        "original_train_count": 4,
        "train_count_used": 3,
        "validation_count": 1,
    }
    assert out["fit_kwargs_used"] == {
        "epochs": 2,
        "lr": 1e-3,
        "batch_size": 8,
        "num_workers": 2,
        "weight_decay": 0.001,
        "optimizer_name": "adamw",
        "optimizer_momentum": 0.8,
        "optimizer_nesterov": True,
        "optimizer_dampening": 0.0,
        "adam_beta1": 0.82,
        "adam_beta2": 0.96,
        "adam_amsgrad": True,
        "optimizer_eps": 1e-6,
        "rmsprop_alpha": 0.95,
        "rmsprop_centered": False,
        "scheduler_name": "plateau",
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-5,
        "scheduler_cooldown": 1,
        "scheduler_threshold": 5e-4,
        "scheduler_threshold_mode": "abs",
        "scheduler_eps": 1e-7,
        "criterion_name": "mae",
        "shuffle_train": False,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.25,
        "validation_inputs": ["d"],
        "max_steps": 5,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.01,
    }
    assert out["timing"]["fit_s"] >= 0.0
    assert out["timing"]["total_s"] >= out["timing"]["fit_s"]


def test_micro_finetune_falls_back_when_kwargs_unsupported():
    class _Detector:
        def __init__(self):
            self.called = False

        def fit(self, X):  # noqa: ANN001 - test stub
            self.called = True
            self.n = len(list(X))
            return self

    det = _Detector()
    out = micro_finetune(det, ["a"], fit_kwargs={"epochs": 2})

    assert det.called is True
    assert det.n == 1
    assert out["fit_kwargs_used"] == {}


def test_micro_finetune_passes_multistep_scheduler_milestones():
    class _Detector:
        def __init__(self):
            self.called = False
            self.received = None

        def fit(
            self,
            X,
            *,
            scheduler_name=None,
            scheduler_milestones=None,
            scheduler_gamma=None,
        ):  # noqa: ANN001 - test stub
            self.called = True
            self.received = {
                "scheduler_name": scheduler_name,
                "scheduler_milestones": list(scheduler_milestones or []),
                "scheduler_gamma": scheduler_gamma,
                "n": len(list(X)),
            }
            return self

    det = _Detector()
    out = micro_finetune(
        det,
        ["a", "b", "c"],
        fit_kwargs={
            "scheduler_name": "multistep",
            "scheduler_milestones": [2, 5],
            "scheduler_gamma": 0.3,
        },
    )

    assert det.called is True
    assert det.received == {
        "scheduler_name": "multistep",
        "scheduler_milestones": [2, 5],
        "scheduler_gamma": 0.3,
        "n": 3,
    }
    assert out["fit_kwargs_used"] == {
        "scheduler_name": "multistep",
        "scheduler_milestones": [2, 5],
        "scheduler_gamma": 0.3,
    }


def test_micro_finetune_passes_ema_kwargs():
    class _Detector:
        def __init__(self):
            self.called = False
            self.received = None

        def fit(self, X, *, ema_enabled=None, ema_decay=None, ema_start_epoch=None):  # noqa: ANN001
            self.called = True
            self.received = {
                "ema_enabled": ema_enabled,
                "ema_decay": ema_decay,
                "ema_start_epoch": ema_start_epoch,
                "n": len(list(X)),
            }
            return self

    det = _Detector()
    out = micro_finetune(
        det,
        ["a", "b"],
        fit_kwargs={
            "ema_enabled": True,
            "ema_decay": 0.996,
            "ema_start_epoch": 2,
        },
    )

    assert det.called is True
    assert det.received == {
        "ema_enabled": True,
        "ema_decay": 0.996,
        "ema_start_epoch": 2,
        "n": 2,
    }
    assert out["fit_kwargs_used"] == {
        "ema_enabled": True,
        "ema_decay": 0.996,
        "ema_start_epoch": 2,
    }


def test_micro_finetune_applies_ema_attr_overrides():
    class _Detector:
        def __init__(self):
            self.called = False
            self.ema_enabled = False
            self.ema_decay = 0.999
            self.ema_start_epoch = 1

        def fit(self, X):  # noqa: ANN001 - test stub
            self.called = True
            self.n = len(list(X))
            return self

    det = _Detector()
    out = micro_finetune(
        det,
        ["a", "b", "c"],
        fit_kwargs={
            "ema_enabled": True,
            "ema_decay": 0.995,
            "ema_start_epoch": 3,
        },
    )

    assert det.called is True
    assert det.n == 3
    assert det.ema_enabled is True
    assert det.ema_decay == pytest.approx(0.995)
    assert det.ema_start_epoch == 3
    assert out["fit_kwargs_used"] == {}
    assert out["detector_attr_overrides_used"] == {
        "ema_enabled": True,
        "ema_decay": 0.995,
        "ema_start_epoch": 3,
    }


def test_micro_finetune_applies_subset_limit_and_attribute_overrides():
    class _Detector:
        def __init__(self):
            self.called = False
            self.epochs = 1
            self.lr = 1e-4
            self.batch_size = 1
            self.num_workers = 0
            self.weight_decay = 0.0
            self.optimizer_name = "adam"
            self.optimizer_momentum = None
            self.optimizer_nesterov = False
            self.optimizer_dampening = None
            self.adam_beta1 = None
            self.adam_beta2 = None
            self.adam_amsgrad = None
            self.optimizer_eps = None
            self.rmsprop_alpha = None
            self.rmsprop_centered = None
            self.scheduler_name = None
            self.scheduler_step_size = None
            self.scheduler_gamma = None
            self.scheduler_patience = None
            self.scheduler_factor = None
            self.scheduler_min_lr = None
            self.scheduler_cooldown = None
            self.scheduler_threshold = None
            self.scheduler_threshold_mode = None
            self.scheduler_eps = None
            self.criterion_name = "mse"
            self.shuffle_train = True
            self.drop_last = False
            self.pin_memory = False
            self.persistent_workers = False
            self.warmup_epochs = None
            self.warmup_start_factor = None
            self.max_steps = None
            self.early_stopping_patience = None
            self.early_stopping_min_delta = None

        def fit(self, X):  # noqa: ANN001 - test stub
            self.called = True
            self.received = list(X)
            return self

    det = _Detector()
    out = micro_finetune(
        det,
        ["a", "b", "c", "d"],
        seed=7,
        fit_kwargs={
            "epochs": 5,
            "lr": 5e-3,
            "batch_size": 6,
            "num_workers": 2,
            "weight_decay": 0.02,
            "optimizer_name": "sgd",
            "optimizer_momentum": 0.75,
            "optimizer_nesterov": True,
            "optimizer_dampening": 0.0,
            "adam_beta1": 0.83,
            "adam_beta2": 0.97,
            "adam_amsgrad": False,
            "optimizer_eps": 1e-6,
            "rmsprop_alpha": 0.94,
            "rmsprop_centered": True,
            "scheduler_name": "plateau",
            "scheduler_patience": 3,
            "scheduler_factor": 0.4,
            "scheduler_min_lr": 1e-5,
            "scheduler_cooldown": 2,
            "scheduler_threshold": 0.002,
            "scheduler_threshold_mode": "rel",
            "scheduler_eps": 2e-7,
            "scheduler_step_size": 4,
            "scheduler_gamma": 0.3,
            "criterion_name": "mae",
            "shuffle_train": False,
            "drop_last": True,
            "pin_memory": True,
            "persistent_workers": True,
            "warmup_epochs": 2,
            "warmup_start_factor": 0.5,
            "max_steps": 4,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.02,
            "max_train_samples": 2,
        },
    )

    assert det.called is True
    assert len(det.received) == 2
    assert set(det.received).issubset({"a", "b", "c", "d"})
    assert det.epochs == 5
    assert det.lr == pytest.approx(5e-3)
    assert det.batch_size == 6
    assert det.num_workers == 2
    assert det.weight_decay == pytest.approx(0.02)
    assert det.optimizer_name == "sgd"
    assert det.optimizer_momentum == pytest.approx(0.75)
    assert det.optimizer_nesterov is True
    assert det.optimizer_dampening == pytest.approx(0.0)
    assert det.adam_beta1 == pytest.approx(0.83)
    assert det.adam_beta2 == pytest.approx(0.97)
    assert det.adam_amsgrad is False
    assert det.optimizer_eps == pytest.approx(1e-6)
    assert det.rmsprop_alpha == pytest.approx(0.94)
    assert det.rmsprop_centered is True
    assert det.scheduler_name == "plateau"
    assert det.scheduler_step_size == 4
    assert det.scheduler_gamma == pytest.approx(0.3)
    assert det.scheduler_patience == 3
    assert det.scheduler_factor == pytest.approx(0.4)
    assert det.scheduler_min_lr == pytest.approx(1e-5)
    assert det.scheduler_cooldown == 2
    assert det.scheduler_threshold == pytest.approx(0.002)
    assert det.scheduler_threshold_mode == "rel"
    assert det.scheduler_eps == pytest.approx(2e-7)
    assert det.criterion_name == "mae"
    assert det.shuffle_train is False
    assert det.drop_last is True
    assert det.pin_memory is True
    assert det.persistent_workers is True
    assert det.warmup_epochs == 2
    assert det.warmup_start_factor == pytest.approx(0.5)
    assert det.max_steps == 4
    assert det.early_stopping_patience == 3
    assert det.early_stopping_min_delta == pytest.approx(0.02)
    assert out["dataset"] == {
        "original_train_count": 4,
        "train_count_used": 2,
        "validation_count": 0,
    }
    assert out["fit_kwargs_used"] == {}
    assert out["detector_attr_overrides_used"] == {
        "epochs": 5,
        "lr": 5e-3,
        "batch_size": 6,
        "num_workers": 2,
        "weight_decay": 0.02,
        "optimizer_name": "sgd",
        "optimizer_momentum": 0.75,
        "optimizer_nesterov": True,
        "optimizer_dampening": 0.0,
        "adam_beta1": 0.83,
        "adam_beta2": 0.97,
        "adam_amsgrad": False,
        "optimizer_eps": 1e-6,
        "rmsprop_alpha": 0.94,
        "rmsprop_centered": True,
        "scheduler_name": "plateau",
        "scheduler_step_size": 4,
        "scheduler_gamma": 0.3,
        "scheduler_patience": 3,
        "scheduler_factor": 0.4,
        "scheduler_min_lr": 1e-5,
        "scheduler_cooldown": 2,
        "scheduler_threshold": 0.002,
        "scheduler_threshold_mode": "rel",
        "scheduler_eps": 2e-7,
        "criterion_name": "mae",
        "shuffle_train": False,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
        "warmup_epochs": 2,
        "warmup_start_factor": 0.5,
        "max_steps": 4,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.02,
    }


def test_micro_finetune_applies_attribute_overrides_for_var_kwargs_detectors():
    class _Detector:
        def __init__(self):
            self.called = False
            self.epochs = 1
            self.lr = 1e-4
            self.batch_size = 1
            self.num_workers = 0
            self.weight_decay = 0.0
            self.optimizer_name = "adam"
            self.optimizer_momentum = None
            self.optimizer_nesterov = False
            self.optimizer_dampening = None
            self.adam_beta1 = None
            self.adam_beta2 = None
            self.adam_amsgrad = None
            self.optimizer_eps = None
            self.rmsprop_alpha = None
            self.rmsprop_centered = None
            self.scheduler_name = None
            self.scheduler_step_size = None
            self.scheduler_gamma = None
            self.scheduler_patience = None
            self.scheduler_factor = None
            self.scheduler_min_lr = None
            self.scheduler_cooldown = None
            self.scheduler_threshold = None
            self.scheduler_threshold_mode = None
            self.scheduler_eps = None
            self.criterion_name = "mse"
            self.shuffle_train = True
            self.drop_last = False
            self.pin_memory = False
            self.persistent_workers = False
            self.warmup_epochs = None
            self.warmup_start_factor = None

        def fit(self, X, **kwargs):  # noqa: ANN001 - test stub
            self.called = True
            self.received = list(X)
            self.received_kwargs = dict(kwargs)
            return self

    det = _Detector()
    out = micro_finetune(
        det,
        ["a", "b", "c"],
        fit_kwargs={
            "epochs": 4,
            "lr": 1e-3,
            "batch_size": 5,
                "num_workers": 3,
                "weight_decay": 0.01,
                "optimizer_name": "adamw",
                "optimizer_momentum": 0.6,
                "optimizer_nesterov": False,
                "optimizer_dampening": 0.1,
                "adam_beta1": 0.81,
                "adam_beta2": 0.95,
                "adam_amsgrad": True,
                "optimizer_eps": 1e-6,
                "rmsprop_alpha": 0.92,
                "rmsprop_centered": False,
                "scheduler_name": "plateau",
                "scheduler_patience": 2,
                "scheduler_factor": 0.6,
                "scheduler_min_lr": 2e-5,
                "scheduler_cooldown": 1,
                "scheduler_threshold": 0.001,
                "scheduler_threshold_mode": "abs",
                "scheduler_eps": 3e-7,
                "scheduler_step_size": 2,
                "scheduler_gamma": 0.5,
                "criterion_name": "mae",
                "shuffle_train": False,
                "drop_last": True,
                "pin_memory": True,
                "persistent_workers": True,
                "warmup_epochs": 3,
                "warmup_start_factor": 0.25,
            },
        )

    assert det.called is True
    assert det.received == ["a", "b", "c"]
    assert det.received_kwargs == {
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 5,
            "num_workers": 3,
            "weight_decay": 0.01,
            "optimizer_name": "adamw",
            "optimizer_momentum": 0.6,
            "optimizer_nesterov": False,
            "optimizer_dampening": 0.1,
            "adam_beta1": 0.81,
            "adam_beta2": 0.95,
            "adam_amsgrad": True,
            "optimizer_eps": 1e-6,
            "rmsprop_alpha": 0.92,
            "rmsprop_centered": False,
            "scheduler_name": "plateau",
            "scheduler_step_size": 2,
            "scheduler_gamma": 0.5,
            "scheduler_patience": 2,
            "scheduler_factor": 0.6,
            "scheduler_min_lr": 2e-5,
            "scheduler_cooldown": 1,
            "scheduler_threshold": 0.001,
            "scheduler_threshold_mode": "abs",
            "scheduler_eps": 3e-7,
            "criterion_name": "mae",
        "shuffle_train": False,
        "drop_last": True,
            "pin_memory": True,
            "persistent_workers": True,
            "warmup_epochs": 3,
            "warmup_start_factor": 0.25,
            }
    assert det.epochs == 4
    assert det.lr == pytest.approx(1e-3)
    assert det.batch_size == 5
    assert det.num_workers == 3
    assert det.weight_decay == pytest.approx(0.01)
    assert det.optimizer_name == "adamw"
    assert det.optimizer_momentum == pytest.approx(0.6)
    assert det.optimizer_nesterov is False
    assert det.optimizer_dampening == pytest.approx(0.1)
    assert det.adam_beta1 == pytest.approx(0.81)
    assert det.adam_beta2 == pytest.approx(0.95)
    assert det.adam_amsgrad is True
    assert det.optimizer_eps == pytest.approx(1e-6)
    assert det.rmsprop_alpha == pytest.approx(0.92)
    assert det.rmsprop_centered is False
    assert det.scheduler_name == "plateau"
    assert det.scheduler_step_size == 2
    assert det.scheduler_gamma == pytest.approx(0.5)
    assert det.scheduler_patience == 2
    assert det.scheduler_factor == pytest.approx(0.6)
    assert det.scheduler_min_lr == pytest.approx(2e-5)
    assert det.scheduler_cooldown == 1
    assert det.scheduler_threshold == pytest.approx(0.001)
    assert det.scheduler_threshold_mode == "abs"
    assert det.scheduler_eps == pytest.approx(3e-7)
    assert det.criterion_name == "mae"
    assert det.shuffle_train is False
    assert det.drop_last is True
    assert det.pin_memory is True
    assert det.persistent_workers is True
    assert det.warmup_epochs == 3
    assert det.warmup_start_factor == pytest.approx(0.25)
    assert out["fit_kwargs_used"] == {
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 5,
        "num_workers": 3,
        "weight_decay": 0.01,
        "optimizer_name": "adamw",
        "optimizer_momentum": 0.6,
        "optimizer_nesterov": False,
        "optimizer_dampening": 0.1,
        "adam_beta1": 0.81,
        "adam_beta2": 0.95,
        "adam_amsgrad": True,
        "optimizer_eps": 1e-6,
        "rmsprop_alpha": 0.92,
        "rmsprop_centered": False,
        "scheduler_name": "plateau",
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "scheduler_patience": 2,
        "scheduler_factor": 0.6,
        "scheduler_min_lr": 2e-5,
        "scheduler_cooldown": 1,
        "scheduler_threshold": 0.001,
        "scheduler_threshold_mode": "abs",
        "scheduler_eps": 3e-7,
        "criterion_name": "mae",
        "shuffle_train": False,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.25,
    }
    assert out["detector_attr_overrides_used"] == {
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 5,
        "num_workers": 3,
        "weight_decay": 0.01,
        "optimizer_name": "adamw",
        "optimizer_momentum": 0.6,
        "optimizer_nesterov": False,
        "optimizer_dampening": 0.1,
        "adam_beta1": 0.81,
        "adam_beta2": 0.95,
        "adam_amsgrad": True,
        "optimizer_eps": 1e-6,
        "rmsprop_alpha": 0.92,
        "rmsprop_centered": False,
        "scheduler_name": "plateau",
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "scheduler_patience": 2,
        "scheduler_factor": 0.6,
        "scheduler_min_lr": 2e-5,
        "scheduler_cooldown": 1,
        "scheduler_threshold": 0.001,
        "scheduler_threshold_mode": "abs",
        "scheduler_eps": 3e-7,
        "criterion_name": "mae",
        "shuffle_train": False,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.25,
    }


def test_micro_finetune_uses_validation_split_seed_for_reproducible_random_holdout():
    class _Detector:
        def fit(self, X, *, validation_inputs=None):  # noqa: ANN001 - test stub
            self.train_inputs = list(X)
            self.validation_inputs = list(validation_inputs or [])
            return self

    def _run(split_seed: int) -> tuple[list[str], list[str]]:
        det = _Detector()
        micro_finetune(
            det,
            ["a", "b", "c", "d", "e"],
            fit_kwargs={
                "validation_fraction": 0.4,
                "validation_split_seed": split_seed,
            },
        )
        return det.train_inputs, det.validation_inputs

    train_a, val_a = _run(7)
    train_b, val_b = _run(7)
    train_c, val_c = _run(9)

    assert train_a == train_b
    assert val_a == val_b
    assert sorted(train_a + val_a) == ["a", "b", "c", "d", "e"]
    assert len(val_a) == 2
    assert (train_c, val_c) != (train_a, val_a)


def test_micro_finetune_emits_callback_lifecycle_and_tracker_metrics():
    class _RecorderCallback:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def on_train_start(self, *, context):  # noqa: ANN001 - test stub
            self.events.append(("start", context["train_count"]))

        def on_epoch_end(self, *, epoch, metrics, context):  # noqa: ANN001 - test stub
            self.events.append(
                (
                    "epoch",
                    {
                        "epoch": int(epoch),
                        "loss": metrics.get("loss"),
                        "lr": metrics.get("lr"),
                        "train_count": context["train_count"],
                    },
                )
            )

        def on_train_end(self, *, report, context):  # noqa: ANN001 - test stub
            self.events.append(
                (
                    "end",
                    {
                        "fit_kwargs_used": dict(report.get("fit_kwargs_used", {})),
                        "train_count": context["train_count"],
                    },
                )
            )

    class _RecorderTracker:
        def __init__(self) -> None:
            self.params: list[dict[str, object]] = []
            self.metrics: list[dict[str, object]] = []
            self.artifacts: list[dict[str, object]] = []
            self.closed = False

        def log_params(self, params):  # noqa: ANN001 - test stub
            self.params.append(dict(params))

        def log_metrics(self, metrics, *, step=None):  # noqa: ANN001 - test stub
            self.metrics.append({"metrics": dict(metrics), "step": step})

        def log_artifact(self, name, artifact):  # noqa: ANN001 - test stub
            self.artifacts.append({"name": str(name), "artifact": dict(artifact)})

        def close(self) -> None:
            self.closed = True

    class _Detector:
        def fit(self, X, *, epochs=None):  # noqa: ANN001 - test stub
            self.train_count = len(list(X))
            self.epochs = epochs
            self.training_loss_history_ = [0.9, 0.6]
            self.training_lr_history_ = [0.01, 0.005]
            return self

    callback = _RecorderCallback()
    tracker = _RecorderTracker()
    out = micro_finetune(
        _Detector(),
        ["a", "b", "c"],
        fit_kwargs={"epochs": 2},
        callbacks=[callback],
        tracker=tracker,
    )

    assert callback.events[0] == ("start", 3)
    assert callback.events[1] == (
        "epoch",
        {"epoch": 1, "loss": 0.9, "lr": 0.01, "train_count": 3},
    )
    assert callback.events[2] == (
        "epoch",
        {"epoch": 2, "loss": 0.6, "lr": 0.005, "train_count": 3},
    )
    assert callback.events[3] == (
        "end",
        {"fit_kwargs_used": {"epochs": 2}, "train_count": 3},
    )
    assert tracker.params[0]["requested_fit_kwargs"] == {"epochs": 2}
    epoch_metrics = [item for item in tracker.metrics if "loss" in item["metrics"]]
    assert epoch_metrics == [
        {"metrics": {"loss": 0.9, "lr": 0.01}, "step": 1},
        {"metrics": {"loss": 0.6, "lr": 0.005}, "step": 2},
    ]
    assert tracker.artifacts[0]["name"] == "training_report.json"
    assert tracker.artifacts[0]["artifact"]["fit_kwargs_used"] == {"epochs": 2}
    assert tracker.closed is True
    assert out["fit_kwargs_used"] == {"epochs": 2}


def test_micro_finetune_reports_exception_to_callbacks_and_closes_tracker():
    class _RecorderCallback:
        def __init__(self) -> None:
            self.errors: list[str] = []

        def on_exception(self, *, error, context):  # noqa: ANN001 - test stub
            self.errors.append(f"{type(error).__name__}:{error}")
            self.train_count = context["train_count"]

    class _RecorderTracker:
        def __init__(self) -> None:
            self.closed = False

        def log_params(self, params):  # noqa: ANN001 - test stub
            self.last_params = dict(params)

        def close(self) -> None:
            self.closed = True

    class _Detector:
        def fit(self, X):  # noqa: ANN001 - test stub
            raise RuntimeError(f"fit failed on {len(list(X))} samples")

    callback = _RecorderCallback()
    tracker = _RecorderTracker()
    with pytest.raises(RuntimeError, match="fit failed on 2 samples"):
        micro_finetune(
            _Detector(),
            ["x", "y"],
            callbacks=[callback],
            tracker=tracker,
        )

    assert callback.errors == ["RuntimeError:fit failed on 2 samples"]
    assert callback.train_count == 2
    assert tracker.closed is True


def test_micro_finetune_resource_profiler_callback_adds_resource_profile():
    from pyimgano.training.callbacks import ResourceProfilingCallback

    class _Detector:
        def fit(self, X):  # noqa: ANN001 - test stub
            self.train_count = len(list(X))
            return self

    out = micro_finetune(
        _Detector(),
        ["a", "b", "c", "d"],
        callbacks=[ResourceProfilingCallback(enable_cuda=False)],
    )

    resource_profile = out.get("resource_profile", {})
    assert resource_profile["train_count"] == 4
    assert resource_profile["duration_s"] >= 0.0
    assert resource_profile["memory"]["peak_bytes"] >= resource_profile["memory"]["current_bytes"]
    assert resource_profile["cuda"]["enabled"] is False
