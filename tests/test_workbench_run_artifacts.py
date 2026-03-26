from __future__ import annotations

import json

import pytest

from pyimgano.workbench.run_artifacts import (
    extract_threshold,
    load_report_from_run,
    load_workbench_config_from_run,
    resolve_checkpoint_path,
    select_category_report,
)


def test_run_artifacts_load_workbench_config_from_run_parses_config_json(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "dataset": {"name": "custom", "root": "/tmp/data"},
                    "model": {"name": "vision_patchcore"},
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_workbench_config_from_run(tmp_path)

    assert cfg.dataset.name == "custom"
    assert cfg.model.name == "vision_patchcore"


def test_run_artifacts_select_category_report_requires_explicit_choice_for_multi_category() -> None:
    report = {"per_category": {"a": {"threshold": 0.1}, "b": {"threshold": 0.2}}}

    with pytest.raises(ValueError, match="please specify --from-run-category"):
        select_category_report(report, category=None)


def test_run_artifacts_helpers_extract_threshold_and_checkpoint_path(tmp_path) -> None:
    payload = {
        "threshold": "0.42",
        "checkpoint": {"path": "checkpoints/model.pt"},
    }

    assert extract_threshold(payload) == pytest.approx(0.42)
    assert resolve_checkpoint_path(tmp_path, payload) == (tmp_path / "checkpoints" / "model.pt")


def test_run_artifacts_load_report_from_run_reads_report_json(tmp_path) -> None:
    expected = {"category": "capsule", "threshold": 0.7}
    (tmp_path / "report.json").write_text(json.dumps(expected), encoding="utf-8")

    report = load_report_from_run(tmp_path)

    assert report == expected
