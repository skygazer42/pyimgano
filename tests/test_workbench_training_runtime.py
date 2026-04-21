from __future__ import annotations

from pathlib import Path

import numpy as np

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.training_runtime import run_workbench_training


def test_workbench_training_runtime_runs_micro_finetune_and_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls["micro_finetune"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "seed": seed,
            "fit_kwargs": dict(fit_kwargs or {}),
        }
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        calls["save_checkpoint"] = {"detector": detector, "path": out}
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module

    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "epochs": 2,
                "lr": 0.001,
                "validation_fraction": 0.25,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.01,
                "max_steps": 7,
                "max_train_samples": 16,
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
                "persistent_workers": False,
                "validation_split_seed": 17,
                "warmup_epochs": 3,
                "warmup_start_factor": 0.25,
                "checkpoint_name": "model.pt",
            },
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_a.png", "train_b.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert calls["micro_finetune"] == {
        "detector": result.detector,
        "train_inputs": ["train_a.png", "train_b.png"],
        "seed": 123,
        "fit_kwargs": {
            "epochs": 2,
            "lr": 0.001,
            "validation_fraction": 0.25,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.01,
            "max_steps": 7,
            "max_train_samples": 16,
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
            "persistent_workers": False,
            "validation_split_seed": 17,
            "warmup_epochs": 3,
            "warmup_start_factor": 0.25,
        },
    }
    assert Path(calls["save_checkpoint"]["path"]).as_posix().endswith("checkpoints/custom/model.pt")
    assert result.training_report == {
        "fit_kwargs_used": {
            "epochs": 2,
            "lr": 0.001,
            "validation_fraction": 0.25,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.01,
            "max_steps": 7,
            "max_train_samples": 16,
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
            "persistent_workers": False,
            "validation_split_seed": 17,
            "warmup_epochs": 3,
            "warmup_start_factor": 0.25,
        }
    }
    assert result.checkpoint_meta == {"path": "checkpoints/custom/model.pt"}


def test_workbench_training_runtime_restores_checkpoint_before_micro_finetune(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []

    def _fake_restore(detector, checkpoint_path):  # noqa: ANN001
        calls.append(("restore", {"detector": detector, "checkpoint_path": str(checkpoint_path)}))

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls.append(
            (
                "micro_finetune",
                {
                    "detector": detector,
                    "train_inputs": list(train_inputs),
                    "seed": seed,
                    "fit_kwargs": dict(fit_kwargs or {}),
                },
            )
        )
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module
    import pyimgano.workbench.checkpoint_restore as checkpoint_restore_module

    monkeypatch.setattr(checkpoint_restore_module, "load_checkpoint_into_detector", _fake_restore)
    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    resume_path = tmp_path / "bootstrap.pt"
    resume_path.write_text("seed", encoding="utf-8")
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 42,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "epochs": 2,
                "resume_from_checkpoint": str(resume_path),
            },
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_a.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert calls == [
        (
            "restore",
            {"detector": result.detector, "checkpoint_path": str(resume_path)},
        ),
        (
            "micro_finetune",
            {
                "detector": result.detector,
                "train_inputs": ["train_a.png"],
                "seed": 42,
                "fit_kwargs": {"epochs": 2},
            },
        ),
    ]
    assert result.training_report == {
        "fit_kwargs_used": {"epochs": 2},
        "checkpoint_restore": {
            "requested_path": str(resume_path),
            "loaded": True,
            "path": str(resume_path),
        },
    }


def test_workbench_training_runtime_passes_multistep_scheduler_kwargs(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls["micro_finetune"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "seed": seed,
            "fit_kwargs": dict(fit_kwargs or {}),
        }
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module

    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 9,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "scheduler_name": "multistep",
                "scheduler_milestones": [2, 4],
                "scheduler_gamma": 0.3,
            },
            "output": {"save_run": True},
        }
    )

    run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_a.png", "train_b.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert calls["micro_finetune"] == {
        "detector": calls["micro_finetune"]["detector"],
        "train_inputs": ["train_a.png", "train_b.png"],
        "seed": 9,
        "fit_kwargs": {
            "scheduler_name": "multistep",
            "scheduler_milestones": [2, 4],
            "scheduler_gamma": 0.3,
        },
    }


def test_workbench_training_runtime_passes_ema_kwargs(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls["micro_finetune"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "seed": seed,
            "fit_kwargs": dict(fit_kwargs or {}),
        }
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module

    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 11,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "ema_enabled": True,
                "ema_decay": 0.997,
                "ema_start_epoch": 2,
            },
            "output": {"save_run": True},
        }
    )

    run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_a.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert calls["micro_finetune"] == {
        "detector": calls["micro_finetune"]["detector"],
        "train_inputs": ["train_a.png"],
        "seed": 11,
        "fit_kwargs": {
            "ema_enabled": True,
            "ema_decay": 0.997,
            "ema_start_epoch": 2,
        },
    }


def test_workbench_training_runtime_falls_back_to_detector_fit(tmp_path: Path) -> None:
    class _DummyDetector:
        def __init__(self) -> None:
            self.fit_inputs = None

        def fit(self, train_inputs):  # noqa: ANN001 - test stub
            self.fit_inputs = list(train_inputs)
            return self

    detector = _DummyDetector()
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    result = run_workbench_training(
        detector=detector,
        train_inputs=["train_only.png"],
        config=cfg,
        category="custom",
        run_dir=None,
    )

    assert detector.fit_inputs == ["train_only.png"]
    assert result.detector is detector
    assert result.training_report is None
    assert result.checkpoint_meta is None


def test_workbench_training_runtime_saves_checkpoint_for_fit_only_runs(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        calls["save_checkpoint"] = {"detector": detector, "path": out}
        return out

    import pyimgano.training.checkpointing as checkpointing_module

    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        def __init__(self) -> None:
            self.fit_inputs = None

        def fit(self, train_inputs):  # noqa: ANN001 - test stub
            self.fit_inputs = list(train_inputs)
            return self

    detector = _DummyDetector()
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=detector,
        train_inputs=["train_only.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert detector.fit_inputs == ["train_only.png"]
    assert Path(calls["save_checkpoint"]["path"]).as_posix().endswith("checkpoints/custom/model.pt")
    assert result.checkpoint_meta == {"path": "checkpoints/custom/model.pt"}


def test_workbench_training_runtime_fit_only_checkpoint_save_is_best_effort(
    monkeypatch, tmp_path: Path
) -> None:
    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        _ = detector, path
        raise NotImplementedError("unsupported detector checkpoint")

    import pyimgano.training.checkpointing as checkpointing_module

    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        def __init__(self) -> None:
            self.fit_inputs = None

        def fit(self, train_inputs):  # noqa: ANN001 - test stub
            self.fit_inputs = list(train_inputs)
            return self

    detector = _DummyDetector()
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=detector,
        train_inputs=["train_only.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert detector.fit_inputs == ["train_only.png"]
    assert result.checkpoint_meta is None


def test_workbench_training_runtime_micro_finetune_checkpoint_save_is_best_effort(
    monkeypatch, tmp_path: Path
) -> None:
    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        _ = train_inputs, seed, fit_kwargs, callbacks, tracker
        return {"trained": True, "detector": detector}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        _ = detector, path
        raise NotImplementedError("unsupported detector checkpoint")

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module

    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {"enabled": True, "epochs": 1},
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_only.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert result.training_report == {"trained": True, "detector": result.detector}
    assert result.checkpoint_meta is None


def test_workbench_training_runtime_skips_custom_anomalydino_checkpoint_save(
    tmp_path: Path,
) -> None:
    from pyimgano.models.anomalydino import VisionAnomalyDINO

    class _FakeEmbedder:
        def embed(self, image):  # noqa: ANN001 - test stub
            _ = image
            return np.zeros((4, 2), dtype=np.float32), (2, 2), (8, 8)

    detector = VisionAnomalyDINO(
        embedder=_FakeEmbedder(),
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
    )
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_anomalydino"},
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=detector,
        train_inputs=["train_only.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert result.checkpoint_meta is None


def test_workbench_training_runtime_passes_tracker_and_callbacks(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class _FakeTracker:
        pass

    fake_tracker = _FakeTracker()

    def _fake_create_training_tracker(**kwargs):  # noqa: ANN001
        calls["create_tracker"] = dict(kwargs)
        return fake_tracker

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls["micro_finetune"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "seed": seed,
            "fit_kwargs": dict(fit_kwargs or {}),
            "callbacks": list(callbacks or []),
            "tracker": tracker,
        }
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module
    import pyimgano.workbench.training_runtime as training_runtime_module

    monkeypatch.setattr(
        training_runtime_module, "create_training_tracker", _fake_create_training_tracker
    )
    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 7,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "epochs": 2,
                "tracker_backend": "jsonl",
                "tracker_dir": str(tmp_path / "tracking"),
                "tracker_project": "pyimgano-dev",
                "tracker_run_name": "unit-run",
                "tracker_mode": "offline",
                "callbacks": ["metrics_logger"],
            },
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_1.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert calls["create_tracker"] == {
        "backend": "jsonl",
        "log_dir": str(tmp_path / "tracking"),
        "project": "pyimgano-dev",
        "run_name": "unit-run",
        "mode": "offline",
    }
    assert calls["micro_finetune"]["tracker"] is fake_tracker
    assert calls["micro_finetune"]["fit_kwargs"] == {"epochs": 2}
    assert len(calls["micro_finetune"]["callbacks"]) == 1
    assert calls["micro_finetune"]["callbacks"][0].__class__.__name__ == "MetricsLoggingCallback"
    assert result.training_report["instrumentation"] == {
        "callbacks": ["metrics_logger"],
        "tracker": {
            "backend": "jsonl",
            "log_dir": str(tmp_path / "tracking"),
            "project": "pyimgano-dev",
            "run_name": "unit-run",
            "mode": "offline",
            "enabled": True,
        },
    }


def test_workbench_training_runtime_builds_resource_profiler_callback(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls["micro_finetune"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "seed": seed,
            "fit_kwargs": dict(fit_kwargs or {}),
            "callbacks": list(callbacks or []),
            "tracker": tracker,
        }
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module

    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 7,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "epochs": 2,
                "callbacks": ["metrics_logger", "resource_profiler"],
            },
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_1.png"],
        config=cfg,
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    callback_names = [cb.__class__.__name__ for cb in calls["micro_finetune"]["callbacks"]]
    assert callback_names == ["MetricsLoggingCallback", "ResourceProfilingCallback"]
    assert calls["micro_finetune"]["tracker"] is None
    assert result.training_report["instrumentation"] == {
        "callbacks": ["metrics_logger", "resource_profiler"],
    }


def test_workbench_training_runtime_resolves_default_mlflow_tracker_dir(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class _FakeTracker:
        pass

    fake_tracker = _FakeTracker()

    def _fake_create_training_tracker(**kwargs):  # noqa: ANN001
        calls["create_tracker"] = dict(kwargs)
        return fake_tracker

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        calls["micro_finetune"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "seed": seed,
            "fit_kwargs": dict(fit_kwargs or {}),
            "callbacks": list(callbacks or []),
            "tracker": tracker,
        }
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_save_checkpoint(detector, path):  # noqa: ANN001
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    import pyimgano.training.checkpointing as checkpointing_module
    import pyimgano.training.runner as training_runner_module
    import pyimgano.workbench.training_runtime as training_runtime_module

    monkeypatch.setattr(
        training_runtime_module, "create_training_tracker", _fake_create_training_tracker
    )
    monkeypatch.setattr(training_runner_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(checkpointing_module, "save_checkpoint", _fake_save_checkpoint)

    class _DummyDetector:
        pass

    run_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 7,
            "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
            "model": {"name": "vision_ecod"},
            "training": {
                "enabled": True,
                "epochs": 2,
                "tracker_backend": "mlflow",
                "tracker_project": "pyimgano-prod",
                "tracker_run_name": "run-42",
            },
            "output": {"save_run": True},
        }
    )

    result = run_workbench_training(
        detector=_DummyDetector(),
        train_inputs=["train_1.png"],
        config=cfg,
        category="custom",
        run_dir=run_dir,
    )

    expected_tracking_dir = str(run_dir / "artifacts" / "tracking")
    assert calls["create_tracker"] == {
        "backend": "mlflow",
        "log_dir": expected_tracking_dir,
        "project": "pyimgano-prod",
        "run_name": "run-42",
        "mode": None,
    }
    assert calls["micro_finetune"]["tracker"] is fake_tracker
    assert result.training_report["instrumentation"] == {
        "callbacks": [],
        "tracker": {
            "backend": "mlflow",
            "log_dir": expected_tracking_dir,
            "project": "pyimgano-prod",
            "run_name": "run-42",
            "mode": None,
            "enabled": True,
        },
    }
