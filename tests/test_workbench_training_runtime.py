from __future__ import annotations

from pathlib import Path

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.training_runtime import run_workbench_training


def test_workbench_training_runtime_runs_micro_finetune_and_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake_micro_finetune(detector, train_inputs, *, seed=None, fit_kwargs=None):  # noqa: ANN001
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
        "fit_kwargs": {"epochs": 2, "lr": 0.001},
    }
    assert str(calls["save_checkpoint"]["path"]).endswith("checkpoints/custom/model.pt")
    assert result.training_report == {"fit_kwargs_used": {"epochs": 2, "lr": 0.001}}
    assert result.checkpoint_meta == {"path": "checkpoints/custom/model.pt"}


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
