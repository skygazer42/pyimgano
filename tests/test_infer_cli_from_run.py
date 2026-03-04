from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_supports_from_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)

    # Minimal workbench config artifact format: {"config": {...}}.
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                    },
                    "output": {"save_run": True, "per_image_jsonl": False, "output_dir": str(run_dir)},
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ckpt_path = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    assert det.loaded == str(ckpt_path)
    assert det.threshold_ == 0.7

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["label"] == 0
    assert second["label"] == 1


def test_infer_cli_from_run_resolves_model_checkpoint_path_relative_to_run_dir(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)
    (run_dir / "artifacts").mkdir(parents=True)

    model_artifact = run_dir / "artifacts" / "backbone.onnx"
    model_artifact.write_text("onnx", encoding="utf-8")

    # Minimal workbench config artifact format: {"config": {...}}.
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_onnx_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                        "checkpoint_path": "artifacts/backbone.onnx",
                    },
                    "output": {"save_run": True, "per_image_jsonl": False, "output_dir": str(run_dir)},
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_onnx_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    seen: dict[str, object] = {}

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

    def _create_model(name, **kwargs):  # noqa: ANN001 - test stub
        seen["name"] = str(name)
        seen["kwargs"] = dict(kwargs)
        return _DummyDetector()

    monkeypatch.setattr(infer_cli, "create_model", _create_model)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    assert seen["name"] == "vision_onnx_ecod"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("checkpoint_path") == str(model_artifact.resolve())
