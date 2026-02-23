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

