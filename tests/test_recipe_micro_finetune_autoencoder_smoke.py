from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.recipes.registry import RECIPE_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig


def test_recipe_micro_finetune_autoencoder_writes_checkpoint(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, x):  # noqa: ANN001
            self.fit_inputs = list(x)
            return self

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_recipe_micro_finetune_autoencoder_dummy_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "micro-finetune-autoencoder",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
            },
            "model": {
                "name": "test_recipe_micro_finetune_autoencoder_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
            },
        }
    )

    # Ensure builtin recipes are registered.
    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("micro-finetune-autoencoder")
    report = recipe(cfg)

    assert Path(report["run_dir"]) == out_dir
    ckpt = out_dir / "checkpoints" / "model.pt"
    assert ckpt.exists()
    assert ckpt.read_text(encoding="utf-8") == "ok"

    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["checkpoint"]["path"].endswith("checkpoints/model.pt")


def test_recipe_micro_finetune_autoencoder_restores_before_training(
    monkeypatch, tmp_path: Path
) -> None:
    import cv2

    restore_calls: list[str] = []
    train_calls: list[dict[str, object]] = []

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    def _fake_restore(*, detector, config):  # noqa: ANN001
        del detector
        restore_calls.append(str(config.training.resume_from_checkpoint))
        return {
            "requested_path": str(config.training.resume_from_checkpoint),
            "loaded": True,
            "path": str(config.training.resume_from_checkpoint),
        }

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        train_calls.append(
            {
                "detector": detector,
                "train_inputs": list(train_inputs),
                "seed": seed,
                "fit_kwargs": dict(fit_kwargs or {}),
            }
        )
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    recipe_module = importlib.import_module("pyimgano.recipes.builtin.micro_finetune_autoencoder")

    monkeypatch.setattr(recipe_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(recipe_module, "restore_training_checkpoint_if_requested", _fake_restore)

    MODEL_REGISTRY.register(
        "test_recipe_micro_finetune_autoencoder_restore_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    resume_path = tmp_path / "bootstrap.pt"
    resume_path.write_text("checkpoint", encoding="utf-8")
    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "micro-finetune-autoencoder",
            "seed": 7,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
            },
            "model": {
                "name": "test_recipe_micro_finetune_autoencoder_restore_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "training": {
                "resume_from_checkpoint": str(resume_path),
                "epochs": 2,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
            },
        }
    )

    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("micro-finetune-autoencoder")
    report = recipe(cfg)

    assert restore_calls == [str(resume_path)]
    assert len(train_calls) == 1
    assert train_calls[0]["train_inputs"] == [
        str(root / "train/normal/train_0.png"),
        str(root / "train/normal/train_1.png"),
    ]
    assert train_calls[0]["seed"] == 7
    assert train_calls[0]["fit_kwargs"] == {"epochs": 2}
    assert train_calls == [
        {
            "detector": train_calls[0]["detector"],
            "train_inputs": [
                str(root / "train/normal/train_0.png"),
                str(root / "train/normal/train_1.png"),
            ],
            "seed": 7,
            "fit_kwargs": {"epochs": 2},
        }
    ]
    assert report["training"]["checkpoint_restore"] == {
        "requested_path": str(resume_path),
        "loaded": True,
        "path": str(resume_path),
    }


def test_recipe_micro_finetune_autoencoder_passes_ema_kwargs(monkeypatch, tmp_path: Path) -> None:
    import cv2

    train_calls: list[dict[str, object]] = []

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        train_calls.append(
            {
                "detector": detector,
                "train_inputs": list(train_inputs),
                "seed": seed,
                "fit_kwargs": dict(fit_kwargs or {}),
            }
        )
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    recipe_module = importlib.import_module("pyimgano.recipes.builtin.micro_finetune_autoencoder")
    monkeypatch.setattr(recipe_module, "micro_finetune", _fake_micro_finetune)

    MODEL_REGISTRY.register(
        "test_recipe_micro_finetune_autoencoder_ema_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "micro-finetune-autoencoder",
            "seed": 5,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
            },
            "model": {
                "name": "test_recipe_micro_finetune_autoencoder_ema_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "training": {
                "ema_enabled": True,
                "ema_decay": 0.996,
                "ema_start_epoch": 2,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
            },
        }
    )

    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("micro-finetune-autoencoder")
    recipe(cfg)

    assert train_calls == [
        {
            "detector": train_calls[0]["detector"],
            "train_inputs": [
                str(root / "train/normal/train_0.png"),
                str(root / "train/normal/train_1.png"),
            ],
            "seed": 5,
            "fit_kwargs": {
                "ema_enabled": True,
                "ema_decay": 0.996,
                "ema_start_epoch": 2,
            },
        }
    ]


def test_recipe_micro_finetune_autoencoder_passes_tracker_and_callbacks(
    monkeypatch, tmp_path: Path
) -> None:
    import cv2

    train_calls: list[dict[str, object]] = []

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    class _FakeTracker:
        pass

    fake_tracker = _FakeTracker()

    def _fake_micro_finetune(
        detector,
        train_inputs,
        *,
        seed=None,
        fit_kwargs=None,
        callbacks=None,
        tracker=None,
    ):  # noqa: ANN001
        train_calls.append(
            {
                "detector": detector,
                "train_inputs": list(train_inputs),
                "seed": seed,
                "fit_kwargs": dict(fit_kwargs or {}),
                "callbacks": list(callbacks or []),
                "tracker": tracker,
            }
        )
        return {"fit_kwargs_used": dict(fit_kwargs or {})}

    def _fake_build_training_tracker(*, config, run_dir):  # noqa: ANN001
        del config, run_dir
        return (
            fake_tracker,
            {
                "backend": "jsonl",
                "log_dir": str(tmp_path / "tracking"),
                "project": "pyimgano-dev",
                "run_name": "recipe-run",
                "mode": "offline",
                "enabled": True,
            },
        )

    recipe_module = importlib.import_module("pyimgano.recipes.builtin.micro_finetune_autoencoder")
    monkeypatch.setattr(recipe_module, "micro_finetune", _fake_micro_finetune)
    monkeypatch.setattr(recipe_module, "build_training_tracker", _fake_build_training_tracker)

    MODEL_REGISTRY.register(
        "test_recipe_micro_finetune_autoencoder_tracker_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "micro-finetune-autoencoder",
            "seed": 5,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
            },
            "model": {
                "name": "test_recipe_micro_finetune_autoencoder_tracker_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "training": {
                "epochs": 2,
                "tracker_backend": "jsonl",
                "tracker_dir": str(tmp_path / "tracking"),
                "tracker_project": "pyimgano-dev",
                "tracker_run_name": "recipe-run",
                "tracker_mode": "offline",
                "callbacks": ["metrics_logger"],
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
            },
        }
    )

    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("micro-finetune-autoencoder")
    report = recipe(cfg)

    assert train_calls[0]["tracker"] is fake_tracker
    assert len(train_calls[0]["callbacks"]) == 1
    assert train_calls[0]["callbacks"][0].__class__.__name__ == "MetricsLoggingCallback"
    assert report["training"]["instrumentation"] == {
        "callbacks": ["metrics_logger"],
        "tracker": {
            "backend": "jsonl",
            "log_dir": str(tmp_path / "tracking"),
            "project": "pyimgano-dev",
            "run_name": "recipe-run",
            "mode": "offline",
            "enabled": True,
        },
    }
