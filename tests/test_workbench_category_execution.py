from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from pyimgano.services.workbench_service import WorkbenchThresholdCalibration
from pyimgano.workbench.category_execution import run_workbench_category
from pyimgano.workbench.config import WorkbenchConfig


def test_run_workbench_category_orchestrates_runtime_boundaries(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.workbench.category_execution as execution_module

    calls: dict[str, object] = {}

    runtime_split = SimpleNamespace(
        train_inputs=["train.png"],
        calibration_inputs=["cal.png"],
        test_inputs=["test.png"],
        test_labels=np.asarray([1], dtype=np.int64),
        test_masks=np.ones((1, 2, 2), dtype=np.uint8),
        pixel_skip_reason="pixel skip",
        test_meta=[{"sample": "test.png"}],
        input_format="rgb_u8_hwc",
    )

    def _fake_load_workbench_split(*, config, category, load_masks):  # noqa: ANN001 - test seam
        calls["load_split"] = {
            "dataset": str(config.dataset.name),
            "category": str(category),
            "load_masks": bool(load_masks),
        }
        return "raw-split"

    def _fake_prepare_workbench_runtime_split(*, config, split):  # noqa: ANN001 - test seam
        calls["prepare_split"] = {
            "dataset": str(config.dataset.name),
            "split": split,
        }
        return runtime_split

    def _fake_build_workbench_runtime_detector(*, config):  # noqa: ANN001 - test seam
        calls["build_detector"] = str(config.model.name)
        return "detector"

    def _fake_run_workbench_training(
        *, detector, train_inputs, config, category, run_dir
    ):  # noqa: ANN001 - test seam
        calls["training"] = {
            "detector": detector,
            "train_inputs": list(train_inputs),
            "category": str(category),
            "run_dir": run_dir,
        }
        return SimpleNamespace(
            detector="trained-detector",
            training_report={"fit_kwargs_used": {"epochs": 2}},
            checkpoint_meta={"path": "checkpoints/custom/model.pt"},
        )

    def _fake_calibrate_workbench_threshold(
        *, detector, calibration_inputs, input_format
    ):  # noqa: ANN001 - test seam
        calls["calibrate"] = {
            "detector": detector,
            "calibration_inputs": list(calibration_inputs),
            "input_format": input_format,
        }
        return WorkbenchThresholdCalibration(
            threshold=0.4,
            quantile=0.9,
            quantile_source="default",
        )

    def _fake_build_postprocess(raw):  # noqa: ANN001 - test seam
        calls["build_postprocess"] = raw
        return "postprocess"

    def _fake_run_workbench_inference(
        *,
        detector,
        test_inputs,
        input_format,
        postprocess,
        save_maps,
        test_labels,
        test_masks,
        threshold,
    ):  # noqa: ANN001 - test seam
        calls["inference"] = {
            "detector": detector,
            "test_inputs": list(test_inputs),
            "input_format": input_format,
            "postprocess": postprocess,
            "save_maps": bool(save_maps),
            "test_labels": np.asarray(test_labels).tolist(),
            "test_masks_shape": None if test_masks is None else list(np.asarray(test_masks).shape),
            "threshold": float(threshold),
        }
        return SimpleNamespace(
            scores=np.asarray([0.8], dtype=np.float32),
            maps=[np.ones((2, 2), dtype=np.float32)],
            eval_results={"threshold": 0.45, "auroc": 0.99},
        )

    def _fake_build_workbench_category_report(*, inputs):  # noqa: ANN001 - test seam
        calls["report"] = {
            "category": inputs.category,
            "train_count": inputs.train_count,
            "calibration_count": inputs.calibration_count,
            "threshold_used": inputs.threshold_used,
            "pixel_skip_reason": inputs.pixel_skip_reason,
            "training_report": inputs.training_report,
            "checkpoint_meta": inputs.checkpoint_meta,
        }
        return {"dataset": "custom", "category": "custom", "results": {"auroc": 0.99}}

    def _fake_save_workbench_category_outputs(
        *, run_dir, outputs, save_maps, per_image_jsonl
    ):  # noqa: ANN001 - test seam
        calls["save_outputs"] = {
            "run_dir": run_dir,
            "payload": dict(outputs.payload),
            "test_inputs": list(outputs.test_inputs),
            "scores": np.asarray(outputs.scores).tolist(),
            "threshold": float(outputs.threshold),
            "save_maps": bool(save_maps),
            "per_image_jsonl": bool(per_image_jsonl),
            "test_meta": list(outputs.test_meta) if outputs.test_meta is not None else None,
        }

    monkeypatch.setattr(execution_module, "load_workbench_split", _fake_load_workbench_split)
    monkeypatch.setattr(
        execution_module,
        "prepare_workbench_runtime_split",
        _fake_prepare_workbench_runtime_split,
    )
    monkeypatch.setattr(
        execution_module,
        "build_workbench_runtime_detector",
        _fake_build_workbench_runtime_detector,
    )
    monkeypatch.setattr(execution_module, "run_workbench_training", _fake_run_workbench_training)
    monkeypatch.setattr(
        execution_module.workbench_service,
        "calibrate_workbench_threshold",
        _fake_calibrate_workbench_threshold,
    )
    monkeypatch.setattr(execution_module, "build_postprocess", _fake_build_postprocess)
    monkeypatch.setattr(execution_module, "run_workbench_inference", _fake_run_workbench_inference)
    monkeypatch.setattr(
        execution_module,
        "build_workbench_category_report",
        _fake_build_workbench_category_report,
    )
    monkeypatch.setattr(
        execution_module,
        "save_workbench_category_outputs",
        _fake_save_workbench_category_outputs,
    )

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "adaptation": {
                "save_maps": True,
                "postprocess": {
                    "normalize": True,
                    "normalize_method": "minmax",
                },
            },
            "output": {
                "output_dir": str(tmp_path / "run_out"),
                "save_run": True,
                "per_image_jsonl": True,
            },
        }
    )

    payload = run_workbench_category(
        config=cfg,
        recipe_name="industrial-adapt",
        category="custom",
        run_dir=tmp_path / "run_out",
    )

    assert payload == {"dataset": "custom", "category": "custom", "results": {"auroc": 0.99}}
    assert calls["load_split"] == {
        "dataset": "custom",
        "category": "custom",
        "load_masks": True,
    }
    assert calls["prepare_split"] == {"dataset": "custom", "split": "raw-split"}
    assert calls["build_detector"] == "vision_ecod"
    assert calls["training"] == {
        "detector": "detector",
        "train_inputs": ["train.png"],
        "category": "custom",
        "run_dir": tmp_path / "run_out",
    }
    assert calls["calibrate"] == {
        "detector": "trained-detector",
        "calibration_inputs": ["cal.png"],
        "input_format": "rgb_u8_hwc",
    }
    assert calls["inference"]["threshold"] == 0.4
    assert calls["report"] == {
        "category": "custom",
        "train_count": 1,
        "calibration_count": 1,
        "threshold_used": 0.45,
        "pixel_skip_reason": "pixel skip",
        "training_report": {"fit_kwargs_used": {"epochs": 2}},
        "checkpoint_meta": {"path": "checkpoints/custom/model.pt"},
    }
    assert calls["save_outputs"] == {
        "run_dir": tmp_path / "run_out",
        "payload": {"dataset": "custom", "category": "custom", "results": {"auroc": 0.99}},
        "test_inputs": ["test.png"],
        "scores": [0.800000011920929],
        "threshold": 0.45,
        "save_maps": True,
        "per_image_jsonl": True,
        "test_meta": [{"sample": "test.png"}],
    }
