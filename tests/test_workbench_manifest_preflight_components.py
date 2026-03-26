from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyimgano.datasets.manifest import ManifestRecord, ManifestSplitPolicy
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.manifest_category_preflight import preflight_manifest_category
from pyimgano.workbench.manifest_record_preflight import (
    load_manifest_records_best_effort,
    resolve_manifest_path_best_effort,
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    lines = [row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_record_preflight_loads_records_and_reports_invalid_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "bottle", "split": "train"},
            "{bad json",
        ],
    )

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    records, categories = load_manifest_records_best_effort(
        manifest_path=manifest,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert len(records) == 1
    assert records[0].category == "bottle"
    assert categories == {"bottle"}
    assert issues[0]["code"] == "MANIFEST_INVALID_JSON"


def test_manifest_record_preflight_resolves_relative_path_from_manifest_dir(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_path = manifest_dir / "img.png"
    image_path.touch()

    resolved, exists, source = resolve_manifest_path_best_effort(
        "img.png",
        manifest_path=manifest_dir / "manifest.jsonl",
        root_fallback=tmp_path / "root",
    )

    assert resolved == str(image_path.resolve())
    assert exists is True
    assert source == "manifest_dir"


def test_manifest_record_preflight_resolves_loaded_records_via_loader(
    monkeypatch, tmp_path: Path
) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_record_preflight")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest record preflight helper: {exc}")

    calls: list[str] = []
    sentinel_records = [SimpleNamespace(category="bottle")]

    def _fake_load_manifest_records_best_effort(
        *, manifest_path, issues, issue_builder
    ):  # noqa: ANN001
        del issues, issue_builder
        calls.append(str(manifest_path))
        return sentinel_records, {"bottle", "cable"}

    monkeypatch.setattr(
        helper_module,
        "load_manifest_records_best_effort",
        _fake_load_manifest_records_best_effort,
    )

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    manifest = tmp_path / "manifest.jsonl"
    result = helper_module.resolve_manifest_preflight_records(
        manifest_path=manifest,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result == {
        "records": sentinel_records,
        "categories": {"bottle", "cable"},
        "summary": None,
    }
    assert issues == []
    assert calls == [str(manifest)]


def test_manifest_record_preflight_returns_error_summary_for_empty_records(tmp_path: Path) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_record_preflight")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest record preflight helper: {exc}")

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, ["# ignored", "", "{bad json"])

    result = helper_module.resolve_manifest_preflight_records(
        manifest_path=manifest,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["records"] == []
    assert result["categories"] == set()
    assert result["summary"] == {
        "manifest_path": str(manifest),
        "manifest": {"ok": False},
    }
    assert [issue["code"] for issue in issues] == ["MANIFEST_INVALID_JSON", "MANIFEST_EMPTY"]


def test_manifest_category_summary_filters_records_and_counts_explicit_splits() -> None:
    try:
        summary_module = importlib.import_module("pyimgano.workbench.manifest_category_summary")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest category summary helper: {exc}")

    records = [
        object(),
        ManifestRecord.from_mapping(
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            lineno=1,
        ),
        ManifestRecord.from_mapping(
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            lineno=2,
        ),
        ManifestRecord.from_mapping(
            {"image_path": "bad.png", "category": "bottle", "split": "test", "label": 1},
            lineno=3,
        ),
    ]

    summary = summary_module.summarize_manifest_category_records(records=records)

    assert len(summary["records"]) == 3
    assert summary["counts"]["total"] == 3
    assert summary["counts"]["explicit_by_split"]["train"] == 1
    assert summary["counts"]["explicit_by_split"]["test"] == 2
    assert summary["counts"]["explicit_test_labels"]["normal"] == 1
    assert summary["counts"]["explicit_test_labels"]["anomaly"] == 1


def test_manifest_source_validation_resolves_valid_manifest_and_warns_on_missing_root(
    tmp_path: Path,
) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_source_validation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest source validation helper: {exc}")

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "bottle", "split": "train"},
        ],
    )
    missing_root = tmp_path / "missing-root"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "manifest",
                "root": str(missing_root),
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

    result = helper_module.resolve_manifest_preflight_source(
        config=cfg,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["summary"] is None
    assert result["manifest_path"] == manifest
    assert result["root_fallback"] == missing_root
    assert [issue["code"] for issue in issues] == ["DATASET_ROOT_MISSING"]


def test_manifest_source_validation_returns_error_summary_when_manifest_path_missing() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_source_validation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest source validation helper: {exc}")

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            input_mode="paths",
            manifest_path=None,
            root=".",
        )
    )
    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    result = helper_module.resolve_manifest_preflight_source(
        config=cfg,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["summary"] == {"manifest": {"ok": False}}
    assert result["manifest_path"] is None
    assert result["root_fallback"] is None
    assert [issue["code"] for issue in issues] == ["MANIFEST_PATH_MISSING"]


def test_manifest_category_selection_expands_all_categories_in_sorted_order() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_category_selection")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest category selection helper: {exc}")

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    result = helper_module.select_manifest_preflight_categories(
        requested_category="all",
        available_categories={"zipper", "bottle", "cable"},
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["requested_all"] is True
    assert result["categories"] == ["bottle", "cable", "zipper"]
    assert issues == []


def test_manifest_category_selection_emits_issue_for_missing_requested_category() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_category_selection")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest category selection helper: {exc}")

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    result = helper_module.select_manifest_preflight_categories(
        requested_category="capsule",
        available_categories={"bottle", "cable"},
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["requested_all"] is False
    assert result["categories"] == ["capsule"]
    assert [issue["code"] for issue in issues] == ["MANIFEST_CATEGORY_EMPTY"]
    assert issues[0]["context"] == {
        "category": "capsule",
        "available_categories": ["bottle", "cable"],
    }


def test_manifest_preflight_report_flattens_single_category_payload() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_preflight_report")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest preflight report helper: {exc}")

    report = helper_module.build_manifest_preflight_report(
        manifest_path="manifest.jsonl",
        root_fallback="dataset-root",
        policy=SimpleNamespace(
            mode="grouped",
            scope="category",
            seed=123,
            test_normal_fraction=0.2,
        ),
        categories=["bottle"],
        per_category={"bottle": {"counts": {"total": 3}}},
        requested_all=False,
    )

    assert report == {
        "manifest_path": "manifest.jsonl",
        "root_fallback": "dataset-root",
        "split_policy": {
            "mode": "grouped",
            "scope": "category",
            "seed": 123,
            "test_normal_fraction": 0.2,
        },
        "categories": ["bottle"],
        "per_category": None,
        "counts": {"total": 3},
        "manifest": {"ok": True},
    }


def test_manifest_preflight_report_keeps_per_category_payload_for_all_request() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_preflight_report")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest preflight report helper: {exc}")

    report = helper_module.build_manifest_preflight_report(
        manifest_path="manifest.jsonl",
        root_fallback=None,
        policy=SimpleNamespace(
            mode="grouped",
            scope="category",
            seed=123,
            test_normal_fraction=0.2,
        ),
        categories=["bottle", "cable"],
        per_category={
            "bottle": {"counts": {"total": 3}},
            "cable": {"counts": {"total": 2}},
        },
        requested_all=True,
    )

    assert report == {
        "manifest_path": "manifest.jsonl",
        "root_fallback": None,
        "split_policy": {
            "mode": "grouped",
            "scope": "category",
            "seed": 123,
            "test_normal_fraction": 0.2,
        },
        "categories": ["bottle", "cable"],
        "per_category": {
            "bottle": {"counts": {"total": 3}},
            "cable": {"counts": {"total": 2}},
        },
        "manifest": {"ok": True},
    }


def test_manifest_split_policy_prefers_explicit_split_policy_seed() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_split_policy")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest split policy helper: {exc}")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "manifest",
                "root": ".",
                "manifest_path": "manifest.jsonl",
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
                "split_policy": {
                    "seed": 77,
                    "mode": "benchmark",
                    "scope": "dataset",
                    "test_normal_fraction": 0.35,
                },
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    policy = helper_module.build_manifest_split_policy(config=cfg)

    assert isinstance(policy, ManifestSplitPolicy)
    assert policy.mode == "benchmark"
    assert policy.scope == "dataset"
    assert policy.seed == 77
    assert policy.test_normal_fraction == pytest.approx(0.35)


def test_manifest_split_policy_falls_back_to_workbench_seed_then_zero() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_split_policy")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest split policy helper: {exc}")

    cfg_with_workbench_seed = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 19,
            "dataset": {
                "name": "manifest",
                "root": ".",
                "manifest_path": "manifest.jsonl",
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
                "split_policy": {"test_normal_fraction": 0.25},
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )
    cfg_without_seed = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "manifest",
                "root": ".",
                "manifest_path": "manifest.jsonl",
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
                "split_policy": {"test_normal_fraction": 0.25},
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"save_run": False},
        }
    )

    with_workbench_seed = helper_module.build_manifest_split_policy(config=cfg_with_workbench_seed)
    without_seed = helper_module.build_manifest_split_policy(config=cfg_without_seed)

    assert with_workbench_seed.seed == 19
    assert without_seed.seed == 0


def test_manifest_preflight_categories_filters_records_and_preserves_order(
    monkeypatch, tmp_path: Path
) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.manifest_preflight_categories")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing manifest preflight categories helper: {exc}")

    calls: list[dict[str, object]] = []

    def _fake_preflight_manifest_category(  # noqa: ANN001 - test seam
        *,
        category,
        records,
        manifest_path,
        root_fallback,
        policy,
        issues,
        issue_builder,
    ):
        del issues, issue_builder
        calls.append(
            {
                "category": str(category),
                "record_categories": [str(record.category) for record in records],
                "manifest_path": str(manifest_path),
                "root_fallback": None if root_fallback is None else str(root_fallback),
                "policy": policy,
            }
        )
        return {"category": str(category), "count": len(records)}

    monkeypatch.setattr(
        helper_module,
        "preflight_manifest_category",
        _fake_preflight_manifest_category,
    )

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    manifest = tmp_path / "manifest.jsonl"
    policy = object()
    per_category = helper_module.preflight_manifest_categories(
        categories=["zipper", "capsule", "bottle"],
        records=[
            SimpleNamespace(category="bottle"),
            SimpleNamespace(category="zipper"),
            SimpleNamespace(category="bottle"),
        ],
        manifest_path=manifest,
        root_fallback=tmp_path,
        policy=policy,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert list(per_category.keys()) == ["zipper", "capsule", "bottle"]
    assert per_category == {
        "zipper": {"category": "zipper", "count": 1},
        "capsule": {"category": "capsule", "count": 0},
        "bottle": {"category": "bottle", "count": 2},
    }
    assert calls == [
        {
            "category": "zipper",
            "record_categories": ["zipper"],
            "manifest_path": str(manifest),
            "root_fallback": str(tmp_path),
            "policy": policy,
        },
        {
            "category": "capsule",
            "record_categories": [],
            "manifest_path": str(manifest),
            "root_fallback": str(tmp_path),
            "policy": policy,
        },
        {
            "category": "bottle",
            "record_categories": ["bottle", "bottle"],
            "manifest_path": str(manifest),
            "root_fallback": str(tmp_path),
            "policy": policy,
        },
    ]


def test_non_manifest_source_validation_returns_error_summary_when_root_missing(
    tmp_path: Path,
) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.non_manifest_source_validation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest source validation helper: {exc}")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "mvtec",
                "root": str(tmp_path / "missing-root"),
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

    result = helper_module.resolve_non_manifest_preflight_source(
        config=cfg,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["dataset"] == "mvtec"
    assert result["root"] == tmp_path / "missing-root"
    assert result["summary"] == {"dataset_root": str(tmp_path / "missing-root"), "ok": False}
    assert [issue["code"] for issue in issues] == ["DATASET_ROOT_MISSING"]


def test_non_manifest_source_validation_emits_custom_structure_issue(tmp_path: Path) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.non_manifest_source_validation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest source validation helper: {exc}")

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
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
    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    result = helper_module.resolve_non_manifest_preflight_source(
        config=cfg,
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["dataset"] == "custom"
    assert result["root"] == root
    assert result["summary"] is None
    assert [issue["code"] for issue in issues] == ["CUSTOM_DATASET_INVALID_STRUCTURE"]


def test_non_manifest_category_selection_expands_all_categories_in_sorted_order() -> None:
    try:
        helper_module = importlib.import_module(
            "pyimgano.workbench.non_manifest_category_selection"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest category selection helper: {exc}")

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    result = helper_module.select_non_manifest_preflight_categories(
        requested_category="all",
        available_categories=["zipper", "bottle", "cable"],
        dataset="mvtec",
        root="/tmp/mvtec",
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["requested_all"] is True
    assert result["categories"] == ["bottle", "cable", "zipper"]
    assert issues == []


def test_non_manifest_category_selection_emits_issue_for_missing_requested_category() -> None:
    try:
        helper_module = importlib.import_module(
            "pyimgano.workbench.non_manifest_category_selection"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest category selection helper: {exc}")

    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    result = helper_module.select_non_manifest_preflight_categories(
        requested_category="capsule",
        available_categories=["bottle", "cable"],
        dataset="mvtec",
        root="/tmp/mvtec",
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result["requested_all"] is False
    assert result["categories"] == ["capsule"]
    assert [issue["code"] for issue in issues] == ["DATASET_CATEGORY_EMPTY"]
    assert issues[0]["context"] == {
        "dataset": "mvtec",
        "root": "/tmp/mvtec",
        "category": "capsule",
        "available_categories": ["bottle", "cable"],
    }


def test_non_manifest_category_listing_returns_categories_from_loader(
    monkeypatch, tmp_path: Path
) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.non_manifest_category_listing")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest category listing helper: {exc}")

    root = tmp_path / "mvtec"
    root.mkdir(parents=True, exist_ok=True)
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
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
    calls: list[str] = []

    def _fake_list_workbench_categories(*, config):  # noqa: ANN001 - test seam
        calls.append(str(config.dataset.name))
        return ["bottle", "capsule"]

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    monkeypatch.setattr(helper_module, "list_workbench_categories", _fake_list_workbench_categories)

    result = helper_module.load_non_manifest_preflight_categories(
        config=cfg,
        dataset="mvtec",
        root=str(root),
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result == {
        "categories": ["bottle", "capsule"],
        "summary": None,
    }
    assert issues == []
    assert calls == ["mvtec"]


def test_non_manifest_category_listing_returns_error_summary_on_loader_failure(
    monkeypatch, tmp_path: Path
) -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.non_manifest_category_listing")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest category listing helper: {exc}")

    root = tmp_path / "mvtec"
    root.mkdir(parents=True, exist_ok=True)
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
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

    def _failing_list_workbench_categories(*, config):  # noqa: ANN001 - test seam
        raise RuntimeError(f"boom:{config.dataset.name}")

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    monkeypatch.setattr(
        helper_module, "list_workbench_categories", _failing_list_workbench_categories
    )

    result = helper_module.load_non_manifest_preflight_categories(
        config=cfg,
        dataset="mvtec",
        root=str(root),
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert result == {
        "categories": None,
        "summary": {"dataset_root": str(root), "ok": False},
    }
    assert [issue["code"] for issue in issues] == ["DATASET_CATEGORY_LIST_FAILED"]
    assert issues[0]["context"] == {
        "dataset": "mvtec",
        "root": str(root),
        "error": "boom:mvtec",
    }


def test_non_manifest_preflight_report_builds_success_summary() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.non_manifest_preflight_report")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing non-manifest preflight report helper: {exc}")

    report = helper_module.build_non_manifest_preflight_report(
        root="dataset-root",
        categories=["bottle", "capsule"],
    )

    assert report == {
        "dataset_root": "dataset-root",
        "categories": ["bottle", "capsule"],
        "ok": True,
    }


def test_manifest_category_preflight_builds_summary_and_pixel_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    (tmp_path / "train.png").touch()
    (tmp_path / "good.png").touch()
    (tmp_path / "bad.png").touch()
    (tmp_path / "bad_mask.png").touch()

    records = [
        ManifestRecord.from_mapping(
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            lineno=1,
        ),
        ManifestRecord.from_mapping(
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            lineno=2,
        ),
        ManifestRecord.from_mapping(
            {
                "image_path": "bad.png",
                "category": "bottle",
                "split": "test",
                "label": 1,
                "mask_path": "bad_mask.png",
            },
            lineno=3,
        ),
    ]
    issues: list[dict[str, object]] = []

    def _issue_builder(code, severity, message, *, context=None):  # noqa: ANN001 - test seam
        return {
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "context": context,
        }

    summary = preflight_manifest_category(
        category="bottle",
        records=records,
        manifest_path=manifest,
        root_fallback=tmp_path,
        policy=ManifestSplitPolicy(seed=123, test_normal_fraction=0.2),
        issues=issues,
        issue_builder=_issue_builder,
    )

    assert summary["counts"]["total"] == 3
    assert summary["assigned_counts"]["train"] == 1
    assert summary["assigned_counts"]["test"] == 2
    assert summary["pixel_metrics"]["enabled"] is True
    assert issues == []
