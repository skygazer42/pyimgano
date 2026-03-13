from __future__ import annotations

import json
from pathlib import Path

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.manifest_preflight import run_manifest_preflight


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_manifest_preflight_returns_expected_summary_shape(tmp_path: Path) -> None:
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

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    summary = run_manifest_preflight(config=cfg, issues=issues, issue_builder=_issue_builder)

    assert summary["manifest"]["ok"] is True
    assert summary["counts"]["total"] == 3
    assert summary["assigned_counts"]["train"] == 1
    assert summary["assigned_counts"]["test"] == 2
    assert summary["pixel_metrics"]["enabled"] is True
    assert issues == []


def test_run_manifest_preflight_uses_split_policy_boundary(monkeypatch, tmp_path: Path) -> None:
    import pyimgano.workbench.manifest_preflight as helper_module
    from pyimgano.datasets.manifest import ManifestSplitPolicy

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

    calls: list[int] = []

    def _fake_build_manifest_split_policy(*, config):  # noqa: ANN001 - test seam
        calls.append(int(config.seed))
        return ManifestSplitPolicy(
            mode="benchmark",
            scope="category",
            seed=777,
            test_normal_fraction=0.45,
        )

    monkeypatch.setattr(helper_module, "build_manifest_split_policy", _fake_build_manifest_split_policy)

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

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    summary = run_manifest_preflight(config=cfg, issues=issues, issue_builder=_issue_builder)

    assert summary["split_policy"]["seed"] == 777
    assert summary["split_policy"]["test_normal_fraction"] == 0.45
    assert calls == [123]


def test_run_manifest_preflight_uses_category_batch_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.workbench.manifest_preflight as helper_module

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "a.png").touch()
    (mdir / "b.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "bottle", "split": "train"},
            {"image_path": "b.png", "category": "cable", "split": "train"},
        ],
    )

    calls: list[dict[str, object]] = []

    def _fake_preflight_manifest_categories(  # noqa: ANN001 - test seam
        *,
        categories,
        records,
        manifest_path,
        root_fallback,
        policy,
        issues,
        issue_builder,
    ):
        calls.append(
            {
                "categories": list(categories),
                "record_count": len(records),
                "manifest_path": str(manifest_path),
                "root_fallback": None if root_fallback is None else str(root_fallback),
                "policy_seed": int(policy.seed),
            }
        )
        return {
            "bottle": {"counts": {"total": 1}},
            "cable": {"counts": {"total": 1}},
        }

    monkeypatch.setattr(
        helper_module,
        "preflight_manifest_categories",
        _fake_preflight_manifest_categories,
    )

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
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    summary = run_manifest_preflight(config=cfg, issues=issues, issue_builder=_issue_builder)

    assert summary["per_category"] == {
        "bottle": {"counts": {"total": 1}},
        "cable": {"counts": {"total": 1}},
    }
    assert calls == [
        {
            "categories": ["bottle", "cable"],
            "record_count": 2,
            "manifest_path": str(manifest),
            "root_fallback": str(tmp_path),
            "policy_seed": 123,
        }
    ]


def test_run_manifest_preflight_uses_record_preflight_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.workbench.manifest_preflight as helper_module

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    calls: list[str] = []

    def _fake_resolve_manifest_preflight_records(*, manifest_path, issues, issue_builder):  # noqa: ANN001
        calls.append(str(manifest_path))
        return {
            "records": [],
            "categories": set(),
            "summary": {"manifest_path": str(manifest_path), "manifest": {"ok": False}},
        }

    monkeypatch.setattr(
        helper_module,
        "resolve_manifest_preflight_records",
        _fake_resolve_manifest_preflight_records,
    )

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
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    summary = run_manifest_preflight(config=cfg, issues=issues, issue_builder=_issue_builder)

    assert summary == {"manifest_path": str(manifest), "manifest": {"ok": False}}
    assert calls == [str(manifest)]
