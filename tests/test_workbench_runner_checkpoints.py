from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def test_workbench_runner_writes_checkpoint_when_training_enabled(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)
            self.fit_calls = 0

        def fit(self, X, *, epochs=None, lr=None):  # noqa: ANN001 - test stub
            self.fit_calls += 1
            self.fit_inputs = list(X)
            self.fit_kwargs = {"epochs": epochs, "lr": lr}
            return self

        def decision_function(self, X):  # noqa: ANN001
            n = len(list(X))
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_workbench_runner_checkpoint_dummy_detector",
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
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
                "limit_test": 2,
            },
            "model": {
                "name": "test_workbench_runner_checkpoint_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "training": {
                "enabled": True,
                "epochs": 2,
                "lr": 0.001,
                "checkpoint_name": "model.pt",
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
                "per_image_jsonl": False,
            },
        }
    )

    report = run_workbench(config=cfg, recipe_name="industrial-adapt")
    assert Path(report["run_dir"]) == out_dir

    ckpt = out_dir / "checkpoints" / "custom" / "model.pt"
    assert ckpt.exists()
    assert ckpt.read_text(encoding="utf-8") == "ok"

    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["checkpoint"]["path"].endswith("checkpoints/custom/model.pt")
    assert report_json["training"]["fit_kwargs_used"] == {"epochs": 2, "lr": 0.001}

