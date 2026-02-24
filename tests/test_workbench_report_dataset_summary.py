from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_workbench_report_includes_dataset_summary(tmp_path: Path) -> None:
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
        "test_workbench_dataset_summary_dummy_detector",
        _DummyDetector,
        tags=("classical",),
        overwrite=True,
    )

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "train.png").touch()
    (mdir / "good.png").touch()
    (mdir / "bad.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            {"image_path": "bad.png", "category": "bottle", "split": "test", "label": 1},
        ],
    )

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "manifest",
                "root": str(tmp_path),
                "manifest_path": str(manifest),
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {
                "name": "test_workbench_dataset_summary_dummy_detector",
                "device": "cpu",
                "pretrained": False,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
                "per_image_jsonl": False,
            },
        }
    )

    run_workbench(config=cfg, recipe_name="industrial-adapt")

    cat_level = json.loads(
        (out_dir / "categories" / "bottle" / "report.json").read_text(encoding="utf-8")
    )
    ds = cat_level["dataset_summary"]
    assert ds["train_count"] == 1
    assert ds["calibration_count"] == 1
    assert ds["test_count"] == 2
    assert ds["test_anomaly_count"] == 1
    assert ds["test_anomaly_ratio"] == 0.5
    assert ds["pixel_metrics"]["enabled"] is False
    assert "mask_path" in str(ds["pixel_metrics"]["reason"]).lower()

