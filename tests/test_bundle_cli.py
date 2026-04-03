from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ALLOWED_JSON_PATHS = {
    "artifacts/infer_config.json",
    "bundle_manifest.json",
    "calibration_card.json",
    "config.json",
    "environment.json",
    "handoff_report.json",
    "infer_config.json",
    "report.json",
}


def _resolve_test_path(root: Path, rel_path: str) -> Path:
    if rel_path not in _ALLOWED_JSON_PATHS:
        raise ValueError(f"Unsupported test json path: {rel_path}")
    root_resolved = root.resolve()
    path = (root_resolved / rel_path).resolve()
    path.relative_to(root_resolved)
    return path


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = _resolve_test_path(root, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_json_rejects_unknown_relative_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported test json path"):
        _write_json(tmp_path, "unexpected.json", {"ok": True})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_png(path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_ready_bundle(
    tmp_path: Path,
    *,
    include_manifest: bool = True,
    supports_pixel_outputs: bool = False,
    runtime_policy: dict | None = None,
) -> Path:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest

    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "deploy_bundle"
    model_name = "vision_ecod"

    _write_json(run_dir, "report.json", {"dataset": "custom", "model": model_name})
    _write_json(run_dir, "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir, "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        run_dir,
        "artifacts/infer_config.json",
        {
            "schema_version": 1,
            "model": {"name": model_name, "model_kwargs": {}},
        },
    )

    _write_json(bundle_dir, "report.json", {"dataset": "custom", "model": model_name})
    _write_json(bundle_dir, "config.json", {"config": {"dataset": "custom"}})
    _write_json(bundle_dir, "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        bundle_dir,
        "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "a" * 64},
            "threshold_context": {
                "scope": ("pixel" if supports_pixel_outputs else "image"),
                "category_count": 1,
            },
            **(
                {
                    "image_threshold": {
                        "threshold": 0.5,
                        "provenance": {"method": "fixed", "source": "test"},
                    }
                }
                if not supports_pixel_outputs
                else {
                    "image_threshold": {
                        "threshold": 0.5,
                        "provenance": {"method": "fixed", "source": "test"},
                    }
                }
            ),
        },
    )
    _write_json(
        bundle_dir,
        "infer_config.json",
        {
            "schema_version": 1,
            "model": {
                "name": model_name,
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
                "model_kwargs": {},
            },
            "threshold": 0.5,
            **(
                {"defects": {"pixel_threshold": 0.5, "mask_format": "png"}}
                if supports_pixel_outputs
                else {}
            ),
            "artifact_quality": {
                "status": "deployable",
                "threshold_scope": ("pixel" if supports_pixel_outputs else "image"),
                "has_threshold_provenance": True,
                "has_split_fingerprint": True,
                "has_prediction_policy": False,
                "has_operator_contract": False,
                "has_deploy_bundle": True,
                "has_bundle_manifest": True,
                "required_bundle_artifacts_present": True,
                "bundle_artifact_roles": {
                    "infer_config": ["infer_config.json"],
                    "report": ["report.json"],
                    "config": ["config.json"],
                    "environment": ["environment.json"],
                    "calibration_card": ["calibration_card.json"],
                },
                "audit_refs": {"calibration_card": "calibration_card.json"},
                "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
            },
        },
    )

    if include_manifest:
        manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
        if runtime_policy is not None:
            manifest["runtime_policy"] = dict(runtime_policy)
        _write_json(
            bundle_dir,
            "bundle_manifest.json",
            manifest,
        )

    return bundle_dir


class _DummyDetector:
    def __init__(self) -> None:
        self.threshold_ = None

    def decision_function(self, x):  # noqa: ANN001 - test stub
        return np.linspace(0.0, 1.0, num=len(list(x)), dtype=np.float32)


class _DummyMapDetector(_DummyDetector):
    def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
        _ = item
        anomaly_map = np.zeros((4, 4), dtype=np.float32)
        anomaly_map[1:3, 1:3] = 1.0
        return anomaly_map


def _patch_infer_main_with_rows(monkeypatch, *, rows: list[dict]) -> None:
    import pyimgano.infer_cli as infer_cli

    def _fake_main(argv: list[str]) -> int:
        args = list(argv)
        results_path = Path(args[args.index("--save-jsonl") + 1])
        _write_jsonl(results_path, [dict(row) for row in rows])
        return 0

    monkeypatch.setattr(infer_cli, "main", _fake_main)


def test_bundle_cli_validate_reports_ready_bundle(tmp_path: Path, capsys) -> None:
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)

    rc = main(["validate", str(bundle_dir), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["tool"] == "pyimgano-bundle"
    assert payload["command"] == "validate"
    assert payload["exit_code"] == 0
    assert payload["ready"] is True
    assert payload["reason_codes"] == []
    assert payload["bundle_manifest"]["present"] is True
    assert payload["bundle_manifest"]["valid"] is True
    assert payload["handoff_report_status"] == "missing"
    assert (
        payload["next_action"]
        == f"pyimgano bundle run {bundle_dir} --image-dir /path/to/images --output-dir ./bundle_run --json"
    )
    assert (
        payload["watch_command"]
        == f"pyimgano bundle watch {bundle_dir} --watch-dir /path/to/inbox --output-dir ./bundle_watch --once --json"
    )
    assert payload["contract"]["bundle_type"] == "cpu-offline-qc"
    assert payload["contract"]["output_contract"]["primary_result_file"] == "results.jsonl"
    assert payload["contract"]["runtime_policy"] == {
        "batch_gates": {
            "max_anomaly_rate": None,
            "max_reject_rate": None,
            "max_error_rate": None,
            "min_processed": None,
        }
    }


def test_bundle_cli_validate_reports_runtime_policy_from_manifest(tmp_path: Path, capsys) -> None:
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(
        tmp_path,
        runtime_policy={
            "batch_gates": {
                "max_anomaly_rate": 0.2,
                "max_reject_rate": None,
                "max_error_rate": 0.01,
                "min_processed": 20,
            }
        },
    )

    rc = main(["validate", str(bundle_dir), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"]["runtime_policy"] == {
        "batch_gates": {
            "max_anomaly_rate": 0.2,
            "max_reject_rate": None,
            "max_error_rate": 0.01,
            "min_processed": 20,
        }
    }


def test_bundle_cli_validate_reports_missing_manifest_reason_code(tmp_path: Path, capsys) -> None:
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path, include_manifest=False)

    rc = main(["validate", str(bundle_dir), "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["tool"] == "pyimgano-bundle"
    assert payload["command"] == "validate"
    assert payload["exit_code"] == 1
    assert payload["ready"] is False
    assert "BUNDLE_MISSING_MANIFEST" in payload["reason_codes"]


def test_bundle_cli_validate_blocks_invalid_handoff_report(tmp_path: Path, capsys) -> None:
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    _write_json(
        bundle_dir,
        "handoff_report.json",
        {
            "schema_version": 1,
            "bundle_type": "cpu-offline-qc",
            "files": {"infer_config": "infer_config.json"},
            "threshold_summary": {"scope": "pixel"},
        },
    )

    rc = main(["validate", str(bundle_dir), "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["handoff_report_status"] == "invalid"
    assert "invalid_handoff_report" in payload["blocking_reasons"]


def test_bundle_cli_text_uses_validate_rendering_helper(monkeypatch, capsys, tmp_path: Path) -> None:
    import pyimgano.bundle_cli as bundle_cli

    monkeypatch.setattr(
        bundle_cli,
        "evaluate_bundle",
        lambda bundle_dir, check_hashes=False: {
            "bundle_dir": str(bundle_dir),
            "status": "ready",
            "ready": True,
            "exit_code": 0,
            "reason_codes": [],
            "contract": {},
        },
    )
    monkeypatch.setattr(
        bundle_cli,
        "bundle_rendering",
        type(
            "_StubBundleRendering",
            (),
            {
                "format_bundle_validate_lines": staticmethod(
                    lambda payload: ["bundle_dir=delegated", "status=ready"]
                ),
                "format_bundle_run_lines": staticmethod(lambda report: []),
            },
        ),
        raising=False,
    )

    rc = bundle_cli.main(["validate", str(tmp_path / "bundle")])

    assert rc == 0
    out = capsys.readouterr().out
    assert "bundle_dir=delegated" in out


def test_bundle_cli_watch_once_processes_stable_backlog_and_writes_artifacts(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    watch_dir = tmp_path / "watch_inbox"
    _write_png(watch_dir / "a.png")
    output_dir = tmp_path / "bundle_watch_run"
    calls: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        calls.append(list(argv))
        args = list(argv)
        results_path = Path(args[args.index("--save-jsonl") + 1])
        _write_jsonl(results_path, [{"score": 0.1, "label": 0}])
        return 0

    monkeypatch.setattr(infer_cli, "main", _fake_main)

    rc = main(
        [
            "watch",
            str(bundle_dir),
            "--watch-dir",
            str(watch_dir),
            "--output-dir",
            str(output_dir),
            "--settle-seconds",
            "0",
            "--once",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "pyimgano-bundle"
    assert payload["command"] == "watch"
    assert payload["status"] == "completed"
    assert payload["watch_dir"] == str(watch_dir)
    assert payload["processed"] == 1
    assert payload["pending"] == 0
    assert payload["error"] == 0
    assert payload["artifacts"]["results_jsonl"] == str(output_dir / "results.jsonl")
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "watch_report.json").exists()
    assert (output_dir / "watch_state.json").exists()
    assert (output_dir / "watch_events.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["id"] == "a.png"
    assert rows[0]["image_path"] == str(watch_dir / "a.png")
    state = json.loads((output_dir / "watch_state.json").read_text(encoding="utf-8"))
    assert state["entries"]["a.png"]["status"] == "processed"
    assert len(calls) == 1


def test_bundle_watch_service_skips_processed_fingerprint_and_reprocesses_changed_file(
    tmp_path: Path,
) -> None:
    from pyimgano.services.bundle_watch_service import BundleWatchRequest, run_bundle_watch_once

    bundle_dir = _make_ready_bundle(tmp_path)
    watch_dir = tmp_path / "watch_inputs"
    image_path = watch_dir / "sample.png"
    _write_png(image_path)
    output_dir = tmp_path / "watch_out"
    calls: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        calls.append(list(argv))
        args = list(argv)
        results_path = Path(args[args.index("--save-jsonl") + 1])
        _write_jsonl(results_path, [{"score": float(len(calls)), "label": 0}])
        return 0

    request = BundleWatchRequest(
        bundle_dir=bundle_dir,
        watch_dir=watch_dir,
        output_dir=output_dir,
        settle_seconds=0.0,
        once=True,
    )

    first = run_bundle_watch_once(request, infer_main_impl=_fake_main)
    second = run_bundle_watch_once(request, infer_main_impl=_fake_main)
    image_path.write_bytes(image_path.read_bytes() + b"\x00")
    third = run_bundle_watch_once(request, infer_main_impl=_fake_main)

    assert first["processed"] == 1
    assert second["processed"] == 0
    assert third["processed"] == 1
    assert len(calls) == 2
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2


def test_bundle_watch_service_does_not_retry_same_failed_fingerprint(tmp_path: Path) -> None:
    from pyimgano.services.bundle_watch_service import BundleWatchRequest, run_bundle_watch_once

    bundle_dir = _make_ready_bundle(tmp_path)
    watch_dir = tmp_path / "watch_inputs"
    image_path = watch_dir / "bad.png"
    _write_png(image_path)
    output_dir = tmp_path / "watch_out"
    calls: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 2

    request = BundleWatchRequest(
        bundle_dir=bundle_dir,
        watch_dir=watch_dir,
        output_dir=output_dir,
        settle_seconds=0.0,
        once=True,
    )

    first = run_bundle_watch_once(request, infer_main_impl=_fake_main)
    second = run_bundle_watch_once(request, infer_main_impl=_fake_main)

    assert first["error"] == 1
    assert second["error"] == 1
    assert len(calls) == 1
    state = json.loads((output_dir / "watch_state.json").read_text(encoding="utf-8"))
    assert state["entries"]["bad.png"]["status"] == "error"


def test_bundle_cli_run_writes_results_and_run_report_for_image_dir(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    input_dir = tmp_path / "inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output_dir = tmp_path / "bundle_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["tool"] == "pyimgano-bundle"
    assert payload["command"] == "run"
    assert payload["exit_code"] == 0
    assert payload["status"] == "completed"
    assert payload["reason_codes"] == []
    assert payload["input_summary"] == {"kind": "image_dir", "count": 2}
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "run_report.json").exists()

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["id"] == "a.png"
    assert rows[0]["image_path"] == str(input_dir / "a.png")
    assert rows[0]["category"] is None
    assert rows[0]["meta"] is None

    written_report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert written_report["schema_version"] == 1
    assert written_report["command"] == "run"
    assert written_report["exit_code"] == 0
    assert written_report["status"] == "completed"
    assert written_report["processed"] == 2
    assert written_report["artifact_digests"]["results_jsonl_sha256"] == _sha256(
        output_dir / "results.jsonl"
    )


def test_bundle_cli_run_reports_passing_batch_gate_summary(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    input_dir = tmp_path / "batch_gate_inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output_dir = tmp_path / "batch_gate_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-anomaly-rate",
            "0.5",
            "--min-processed",
            "2",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["batch_verdict"] == "pass"
    assert payload["batch_gate_reason_codes"] == []
    assert payload["reason_codes"] == []
    assert payload["batch_gate_summary"] == {
        "requested": True,
        "evaluated": True,
        "processed": 2,
        "counts": {
            "normal": 1,
            "anomalous": 1,
            "rejected": 0,
            "error": 0,
        },
        "rates": {
            "anomaly_rate": 0.5,
            "reject_rate": 0.0,
            "error_rate": 0.0,
        },
        "thresholds": {
            "max_anomaly_rate": 0.5,
            "max_reject_rate": None,
            "max_error_rate": None,
            "min_processed": 2,
        },
        "sources": {
            "max_anomaly_rate": "cli",
            "max_reject_rate": "unset",
            "max_error_rate": "unset",
            "min_processed": "cli",
        },
        "failed_gates": [],
    }


def test_rewrite_results_jsonl_rejects_path_outside_output_dir(tmp_path: Path) -> None:
    from pyimgano.bundle_cli import _rewrite_results_jsonl

    output_dir = tmp_path / "bundle_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = tmp_path / "outside.jsonl"
    results_path.write_text('{"score": 0.1, "label": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="results.jsonl path must stay within output_dir"):
        _rewrite_results_jsonl(
            output_dir=output_dir,
            results_path=results_path,
            input_records=[{"id": "a.png", "image_path": "a.png"}],
        )


def test_bundle_cli_run_blocks_when_batch_anomaly_rate_exceeds_gate(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    input_dir = tmp_path / "batch_block_inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output_dir = tmp_path / "batch_block_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-anomaly-rate",
            "0.49",
            "--json",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["exit_code"] == 1
    assert payload["batch_verdict"] == "blocked"
    assert payload["batch_gate_reason_codes"] == ["RUN_BATCH_ANOMALY_RATE_EXCEEDED"]
    assert payload["reason_codes"] == ["RUN_BATCH_ANOMALY_RATE_EXCEEDED"]
    assert payload["processed"] == 2
    assert payload["batch_gate_summary"]["failed_gates"] == ["max_anomaly_rate"]
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "run_report.json").exists()


def test_bundle_cli_run_blocks_when_reject_error_or_processed_batch_gates_fail(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    input_dir = tmp_path / "batch_multi_gate_inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    _write_png(input_dir / "c.png")
    output_dir = tmp_path / "batch_multi_gate_run"

    _patch_infer_main_with_rows(
        monkeypatch,
        rows=[
            {"score": 0.01, "label": 0},
            {"score": 0.4, "label": -2, "rejected": True},
            {"status": "error", "stage": "infer", "error": "synthetic failure"},
        ],
    )

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-reject-rate",
            "0.2",
            "--max-error-rate",
            "0.2",
            "--min-processed",
            "4",
            "--json",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["batch_verdict"] == "blocked"
    assert payload["batch_gate_reason_codes"] == [
        "RUN_BATCH_MIN_PROCESSED_NOT_MET",
        "RUN_BATCH_REJECT_RATE_EXCEEDED",
        "RUN_BATCH_ERROR_RATE_EXCEEDED",
    ]
    assert payload["reason_codes"] == payload["batch_gate_reason_codes"]
    assert payload["batch_gate_summary"] == {
        "requested": True,
        "evaluated": True,
        "processed": 3,
        "counts": {
            "normal": 1,
            "anomalous": 0,
            "rejected": 1,
            "error": 1,
        },
        "rates": {
            "anomaly_rate": 0.0,
            "reject_rate": 1 / 3,
            "error_rate": 1 / 3,
        },
        "thresholds": {
            "max_anomaly_rate": None,
            "max_reject_rate": 0.2,
            "max_error_rate": 0.2,
            "min_processed": 4,
        },
        "sources": {
            "max_anomaly_rate": "unset",
            "max_reject_rate": "cli",
            "max_error_rate": "cli",
            "min_processed": "cli",
        },
        "failed_gates": ["min_processed", "max_reject_rate", "max_error_rate"],
    }


def test_bundle_cli_run_uses_manifest_default_batch_gates_and_reports_sources(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(
        tmp_path,
        runtime_policy={
            "batch_gates": {
                "max_anomaly_rate": 0.49,
                "max_reject_rate": None,
                "max_error_rate": None,
                "min_processed": None,
            }
        },
    )
    input_dir = tmp_path / "manifest_default_gate_inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output_dir = tmp_path / "manifest_default_gate_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["batch_verdict"] == "blocked"
    assert payload["batch_gate_reason_codes"] == ["RUN_BATCH_ANOMALY_RATE_EXCEEDED"]
    assert payload["batch_gate_summary"]["thresholds"] == {
        "max_anomaly_rate": 0.49,
        "max_reject_rate": None,
        "max_error_rate": None,
        "min_processed": None,
    }
    assert payload["batch_gate_summary"]["sources"] == {
        "max_anomaly_rate": "bundle_manifest",
        "max_reject_rate": "unset",
        "max_error_rate": "unset",
        "min_processed": "unset",
    }


def test_bundle_cli_run_cli_batch_gates_override_manifest_defaults(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(
        tmp_path,
        runtime_policy={
            "batch_gates": {
                "max_anomaly_rate": 0.49,
                "max_reject_rate": None,
                "max_error_rate": None,
                "min_processed": 2,
            }
        },
    )
    input_dir = tmp_path / "manifest_override_gate_inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output_dir = tmp_path / "manifest_override_gate_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-anomaly-rate",
            "0.5",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["batch_verdict"] == "pass"
    assert payload["batch_gate_summary"]["thresholds"] == {
        "max_anomaly_rate": 0.5,
        "max_reject_rate": None,
        "max_error_rate": None,
        "min_processed": 2,
    }
    assert payload["batch_gate_summary"]["sources"] == {
        "max_anomaly_rate": "cli",
        "max_reject_rate": "unset",
        "max_error_rate": "unset",
        "min_processed": "bundle_manifest",
    }


def test_bundle_cli_run_input_manifest_preserves_contract_fields(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path)
    source_dir = tmp_path / "manifest_inputs"
    _write_png(source_dir / "a.png")
    _write_png(source_dir / "b.png")
    manifest_path = source_dir / "input_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "id": "sample-a",
                "image_path": "a.png",
                "category": "bottle",
                "meta": {"station": "L1"},
            },
            {
                "id": "sample-b",
                "image_path": "b.png",
                "category": "bottle",
                "meta": {"station": "L2"},
            },
        ],
    )
    output_dir = tmp_path / "bundle_manifest_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--input-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input_summary"] == {"kind": "input_manifest.jsonl", "count": 2}

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["id"] == "sample-a"
    assert rows[0]["image_path"] == "a.png"
    assert rows[0]["category"] == "bottle"
    assert rows[0]["meta"] == {"station": "L1"}


def test_bundle_cli_rejects_manifest_image_paths_outside_manifest_root(tmp_path: Path) -> None:
    from pyimgano.bundle_cli import _resolve_manifest_image_path

    source_dir = tmp_path / "manifest_inputs"
    outside_dir = tmp_path / "outside"
    _write_png(outside_dir / "evil.png")
    manifest_path = source_dir / "input_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "id": "sample-evil",
                "image_path": "../outside/evil.png",
            }
        ],
    )
    with pytest.raises((FileNotFoundError, ValueError), match="image_path|path traversal"):
        _resolve_manifest_image_path("../outside/evil.png", manifest_path=manifest_path)


def test_bundle_cli_run_rejects_pixel_exports_when_bundle_contract_disallows_them(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path, supports_pixel_outputs=False)
    input_dir = tmp_path / "inputs"
    _write_png(input_dir / "a.png")
    output_dir = tmp_path / "bundle_run_blocked"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyMapDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-masks",
            "--json",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["command"] == "run"
    assert payload["exit_code"] == 1
    assert payload["status"] == "blocked"
    assert "RUN_PIXEL_OUTPUTS_NOT_SUPPORTED" in payload["reason_codes"]
    assert (output_dir / "run_report.json").exists()
    assert not (output_dir / "results.jsonl").exists()


def test_bundle_cli_run_exports_pixel_outputs_when_bundle_contract_allows_them(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.bundle_cli import main

    bundle_dir = _make_ready_bundle(tmp_path, supports_pixel_outputs=True)
    input_dir = tmp_path / "pixel_inputs"
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output_dir = tmp_path / "pixel_bundle_run"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: _DummyMapDetector())

    rc = main(
        [
            "run",
            str(bundle_dir),
            "--image-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-masks",
            "--export-overlays",
            "--export-defects-regions",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["command"] == "run"
    assert payload["exit_code"] == 0
    assert payload["status"] == "completed"
    assert payload["reason_codes"] == []
    assert payload["artifacts"]["masks_dir"] == str(output_dir / "masks")
    assert payload["artifacts"]["overlays_dir"] == str(output_dir / "overlays")
    assert payload["artifacts"]["defects_regions_jsonl"] == str(
        output_dir / "defects_regions.jsonl"
    )

    saved_masks = sorted((output_dir / "masks").glob("*.png"))
    saved_overlays = sorted((output_dir / "overlays").glob("*.png"))
    assert len(saved_masks) == 2
    assert len(saved_overlays) == 2

    defects_rows = [
        json.loads(line)
        for line in (output_dir / "defects_regions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(defects_rows) == 2

    written_report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert written_report["artifact_digests"]["results_jsonl_sha256"] == _sha256(
        output_dir / "results.jsonl"
    )
    assert written_report["artifact_digests"]["defects_regions_jsonl_sha256"] == _sha256(
        output_dir / "defects_regions.jsonl"
    )
    assert written_report["artifact_digests"]["masks_tree_sha256"]
    assert written_report["artifact_digests"]["overlays_tree_sha256"]
