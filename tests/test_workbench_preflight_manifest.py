from __future__ import annotations

import json
from pathlib import Path

from pyimgano.workbench.config import WorkbenchConfig


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_workbench_preflight_manifest_returns_report(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "train.png").touch()
    (mdir / "good.png").touch()
    (mdir / "bad.png").touch()
    (mdir / "bad_mask.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            {
                "image_path": "bad.png",
                "category": "bottle",
                "split": "test",
                "label": 1,
                "mask_path": "bad_mask.png",
            },
        ],
    )

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
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    assert report.dataset == "manifest"
    assert report.category == "bottle"
    assert isinstance(report.summary, dict)
    assert isinstance(report.issues, list)

    # Basic summary invariants.
    assert report.summary["counts"]["total"] == 3
    assert report.summary["counts"]["explicit_by_split"]["train"] == 1
    assert report.summary["counts"]["explicit_by_split"]["test"] == 2
    assert report.summary["counts"]["explicit_test_labels"]["anomaly"] == 1
    assert report.summary["assigned_counts"]["train"] == 1
    assert report.summary["assigned_counts"]["test"] == 2
    assert report.summary["mask_coverage"]["anomaly_test_total"] == 1
    assert report.summary["mask_coverage"]["anomaly_test_mask_exists"] == 1
    assert report.summary["pixel_metrics"]["enabled"] is True


def test_workbench_preflight_manifest_detects_missing_files_and_duplicates(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "a.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "bottle", "split": "train"},
            {"image_path": "a.png", "category": "bottle", "split": "test", "label": 0},
            {"image_path": "missing.png", "category": "bottle", "split": "test", "label": 1},
        ],
    )

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
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    codes = {i.code for i in report.issues}
    assert "MANIFEST_DUPLICATE_IMAGE" in codes
    assert "MANIFEST_MISSING_IMAGE" in codes


def test_workbench_preflight_manifest_detects_group_conflict(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "a.png").touch()
    (mdir / "b.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "bottle", "split": "train", "group_id": "g1"},
            {"image_path": "b.png", "category": "bottle", "split": "test", "label": 0, "group_id": "g1"},
        ],
    )

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
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    codes = {i.code for i in report.issues}
    assert "MANIFEST_GROUP_SPLIT_CONFLICT" in codes


def test_workbench_preflight_manifest_detects_anomaly_in_train_group(tmp_path: Path) -> None:
    from pyimgano.workbench.preflight import run_preflight

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "a.png").touch()
    (mdir / "b.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "bottle", "split": "train", "group_id": "g2"},
            {"image_path": "b.png", "category": "bottle", "label": 1, "group_id": "g2"},
        ],
    )

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
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    report = run_preflight(config=cfg)
    codes = {i.code for i in report.issues}
    assert "MANIFEST_GROUP_ANOMALY_IN_TRAIN" in codes
