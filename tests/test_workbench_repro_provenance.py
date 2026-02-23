import json
from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def test_workbench_runner_persists_seed_in_report_and_config(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            n = len(list(X))
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_workbench_repro_dummy_detector",
        _DummyDetector,
        tags=("classical",),
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
                "name": "test_workbench_repro_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
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

    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    config_json = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert report_json["seed"] == 123
    assert config_json["config"]["seed"] == 123

