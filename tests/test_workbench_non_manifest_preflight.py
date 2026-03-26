from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.non_manifest_preflight import run_non_manifest_preflight


def test_preflight_summary_dispatches_manifest_configs(monkeypatch, tmp_path: Path) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.preflight_summary")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing preflight summary helper: {exc}")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

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

    calls: list[str] = []

    def _fake_run_manifest_preflight(*, config, issues, issue_builder):  # noqa: ANN001 - test seam
        del issues, issue_builder
        calls.append(str(config.dataset.name))
        return {"summary_source": "manifest"}

    monkeypatch.setattr(helper_module, "run_manifest_preflight", _fake_run_manifest_preflight)

    result = helper_module.resolve_workbench_preflight_summary(
        config=cfg,
        issues=[],
        issue_builder=lambda *args, **kwargs: None,
    )

    assert result == {"summary_source": "manifest"}
    assert calls == ["manifest"]


def test_preflight_summary_dispatches_non_manifest_configs(monkeypatch, tmp_path: Path) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.preflight_summary")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing preflight summary helper: {exc}")

    root = tmp_path / "mvtec"
    root.mkdir(parents=True, exist_ok=True)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "mvtec",
                "root": str(root),
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    calls: list[str] = []

    def _fake_run_non_manifest_preflight(
        *, config, issues, issue_builder
    ):  # noqa: ANN001 - test seam
        del issues, issue_builder
        calls.append(str(config.dataset.name))
        return {"summary_source": "non_manifest"}

    monkeypatch.setattr(
        helper_module,
        "run_non_manifest_preflight",
        _fake_run_non_manifest_preflight,
    )

    result = helper_module.resolve_workbench_preflight_summary(
        config=cfg,
        issues=[],
        issue_builder=lambda *args, **kwargs: None,
    )

    assert result == {"summary_source": "non_manifest"}
    assert calls == ["mvtec"]


def test_preflight_types_module_exposes_public_dataclasses() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.preflight_types")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing preflight types helper: {exc}")

    issue = helper_module.PreflightIssue(
        code="X001",
        severity="error",
        message="bad input",
        context={"field": "dataset.root"},
    )
    report = helper_module.PreflightReport(
        dataset="manifest",
        category="bottle",
        summary={"manifest": {"ok": False}},
        issues=[issue],
    )

    assert helper_module.IssueSeverity.__args__ == ("error", "warning", "info")
    assert issue.code == "X001"
    assert issue.context == {"field": "dataset.root"}
    assert report.dataset == "manifest"
    assert report.issues == [issue]


def test_preflight_issue_factory_builds_preflight_issue() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.preflight_issue_factory")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing preflight issue factory helper: {exc}")

    issue = helper_module.build_preflight_issue(
        code=101,
        severity="warning",
        message=RuntimeError("bad input"),
        context={"field": "dataset.root"},
    )

    assert issue.code == "101"
    assert issue.severity == "warning"
    assert issue.message == "bad input"
    assert issue.context == {"field": "dataset.root"}


def test_preflight_report_builder_builds_preflight_report() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.preflight_report")
        types_module = importlib.import_module("pyimgano.workbench.preflight_types")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing preflight report helper: {exc}")

    issue = types_module.PreflightIssue(
        code="P001",
        severity="warning",
        message="check dataset",
        context={"dataset": "manifest"},
    )

    report = helper_module.build_preflight_report(
        dataset="manifest",
        category="bottle",
        summary={"manifest": {"ok": True}},
        issues=[issue],
    )

    assert report == types_module.PreflightReport(
        dataset="manifest",
        category="bottle",
        summary={"manifest": {"ok": True}},
        issues=[issue],
    )


def test_run_non_manifest_preflight_returns_expected_summary_shape(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.workbench.non_manifest_category_listing as listing_module

    root = tmp_path / "mvtec"
    root.mkdir(parents=True, exist_ok=True)

    calls: list[str] = []

    def _fake_list_workbench_categories(*, config):  # noqa: ANN001 - test seam
        calls.append(str(config.dataset.name))
        return ["bottle", "capsule"]

    monkeypatch.setattr(
        listing_module,
        "list_workbench_categories",
        _fake_list_workbench_categories,
    )

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "mvtec",
                "root": str(root),
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

    summary = run_non_manifest_preflight(config=cfg, issues=issues, issue_builder=_issue_builder)

    assert summary == {
        "dataset_root": str(root),
        "categories": ["bottle", "capsule"],
        "ok": True,
    }
    assert issues == []
    assert calls == ["mvtec"]
