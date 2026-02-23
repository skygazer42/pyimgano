from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def _write_rgb(path: Path, *, size: tuple[int, int] = (16, 16), value: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _write_mask(path: Path, *, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", size, color=0)
    # Simple non-empty defect region.
    for y in range(4, 8):
        for x in range(3, 9):
            img.putpixel((x, y), 255)
    img.save(path)


def test_workbench_manifest_dataset_runs_all_categories(tmp_path: Path) -> None:
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
        "test_workbench_manifest_dummy_detector",
        _DummyDetector,
        tags=("vision", "pixel_map"),
        overwrite=True,
    )

    mdir = tmp_path / "manifest_dir"
    manifest = mdir / "manifest.jsonl"

    rows: list[dict] = []
    for cat in ["bottle", "cable"]:
        # Two normals (auto split ensures one train / one test when fraction=0.5).
        for i in range(2):
            name = f"{cat}_n{i}.png"
            _write_rgb(mdir / name, value=100 + i)
            rows.append({"image_path": name, "category": cat})

        # One anomaly.
        aname = f"{cat}_a0.png"
        _write_rgb(mdir / aname, value=240)
        row: dict = {"image_path": aname, "category": cat, "label": 1}

        # Only "bottle" has masks → "cable" should skip pixel metrics with reason.
        if cat == "bottle":
            mname = f"{cat}_a0_mask.png"
            _write_mask(mdir / mname)
            row["mask_path"] = mname
        rows.append(row)

    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "manifest",
                "root": str(tmp_path),
                "manifest_path": str(manifest),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "split_policy": {"test_normal_fraction": 0.5, "seed": 123},
            },
            "model": {
                "name": "test_workbench_manifest_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "adaptation": {"save_maps": True},
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
                "per_image_jsonl": True,
            },
        }
    )

    report = run_workbench(config=cfg, recipe_name="industrial-adapt")
    assert Path(report["run_dir"]) == out_dir
    assert report["category"] == "all"
    assert report["categories"] == ["bottle", "cable"]

    per_category = report["per_category"]
    assert "bottle" in per_category
    assert "cable" in per_category

    bottle = per_category["bottle"]
    assert "pixel_metrics" in bottle["results"]

    cable = per_category["cable"]
    assert cable.get("pixel_metrics_status", {}).get("enabled") is False
    assert "Missing mask_path" in str(cable.get("pixel_metrics_status", {}).get("reason"))

    # Artifacts: per-image JSONL exists per category.
    for cat in ["bottle", "cable"]:
        p = out_dir / "categories" / cat / "per_image.jsonl"
        assert p.exists()
        first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
        assert first["category"] == cat

