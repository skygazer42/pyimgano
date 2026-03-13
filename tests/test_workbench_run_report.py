from __future__ import annotations

import json
from pathlib import Path

from pyimgano.reporting.runs import build_workbench_run_paths
from pyimgano.workbench.run_report import persist_workbench_run_report


def test_workbench_run_report_persists_payload_with_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = build_workbench_run_paths(run_dir)

    payload = persist_workbench_run_report(
        payload={"dataset": "custom", "category": "custom", "threshold": 0.5},
        paths=paths,
    )

    assert payload["run_dir"] == str(run_dir)
    saved = json.loads(paths.report_json.read_text(encoding="utf-8"))
    assert saved["dataset"] == "custom"
    assert saved["run_dir"] == str(run_dir)


def test_workbench_run_report_returns_payload_unchanged_without_paths() -> None:
    payload = persist_workbench_run_report(
        payload={"dataset": "custom", "category": "custom", "threshold": 0.5},
        paths=None,
    )

    assert payload == {"dataset": "custom", "category": "custom", "threshold": 0.5}
