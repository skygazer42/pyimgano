from __future__ import annotations

from pathlib import Path

from pyimgano.workbench.config import WorkbenchConfig


def test_workbench_preflight_non_manifest_root_missing(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "mvtec",
                "root": str(tmp_path / "missing_root"),
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    codes = {i.code for i in report.issues}
    assert "DATASET_ROOT_MISSING" in codes


def test_workbench_preflight_non_manifest_invalid_category(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    root = tmp_path / "mvtec"
    root.mkdir(parents=True, exist_ok=True)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "mvtec",
                "root": str(root),
                "category": "not_a_category",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    codes = {i.code for i in report.issues}
    assert "DATASET_CATEGORY_EMPTY" in codes


def test_workbench_preflight_custom_dataset_structure_validation(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)

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
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    codes = {i.code for i in report.issues}
    assert "CUSTOM_DATASET_INVALID_STRUCTURE" in codes

