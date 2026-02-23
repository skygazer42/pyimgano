from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.reporting.report import REPORT_SCHEMA_VERSION
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def test_workbench_reports_are_stamped_with_schema_version(tmp_path):
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
        "test_workbench_schema_dummy_detector",
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
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
                "limit_test": 2,
            },
            "model": {
                "name": "test_workbench_schema_dummy_detector",
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
    assert str(out_dir) == report["run_dir"]

    top_level = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    cat_level = json.loads(
        (out_dir / "categories" / "custom" / "report.json").read_text(encoding="utf-8")
    )

    for payload in (top_level, cat_level):
        assert payload["schema_version"] == int(REPORT_SCHEMA_VERSION)
        assert isinstance(payload["timestamp_utc"], str)
        datetime.fromisoformat(payload["timestamp_utc"])

        from pyimgano import __version__ as pyimgano_version

        assert payload["pyimgano_version"] == pyimgano_version

