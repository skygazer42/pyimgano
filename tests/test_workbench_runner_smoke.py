import json
from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def test_workbench_runner_smoke_writes_artifacts(tmp_path):
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

        def predict_anomaly_map(self, X):  # noqa: ANN001
            n = len(list(X))
            return np.zeros((n, 16, 16), dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_workbench_runner_dummy_detector",
        _DummyDetector,
        tags=("vision", "pixel_map"),
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
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
                "limit_test": 2,
            },
            "model": {
                "name": "test_workbench_runner_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "adaptation": {
                "save_maps": True,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
                "per_image_jsonl": True,
            },
        }
    )

    report = run_workbench(config=cfg, recipe_name="industrial-adapt")
    assert Path(report["run_dir"]) == out_dir
    assert (out_dir / "report.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "environment.json").exists()
    assert (out_dir / "categories" / "custom" / "per_image.jsonl").exists()

    maps_dir = out_dir / "artifacts" / "maps"
    assert maps_dir.exists()
    assert any(p.suffix == ".npy" for p in maps_dir.iterdir())

    records = (out_dir / "categories" / "custom" / "per_image.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert records
    first = json.loads(records[0])
    assert "anomaly_map" in first

