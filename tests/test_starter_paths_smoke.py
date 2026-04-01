from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_first_run_command_chain_smoke(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main
    from pyimgano.demo_cli import main as demo_main
    from pyimgano.doctor_cli import main as doctor_main
    from pyimgano.infer_cli import main as infer_main
    from pyimgano.runs_cli import main as runs_main

    rc = doctor_main(["--json", "--profile", "first-run"])
    assert rc == 0
    first_run_payload = json.loads(capsys.readouterr().out)
    assert first_run_payload["workflow_profile"]["profile"] == "first-run"

    dataset_root = tmp_path / "demo_dataset"
    demo_run = tmp_path / "demo_suite"
    summary_path = tmp_path / "demo_summary.json"
    rc = demo_main(
        [
            "--smoke",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(demo_run),
            "--summary-json",
            str(summary_path),
            "--no-pretrained",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = doctor_main(["--json", "--profile", "benchmark", "--dataset-target", str(dataset_root)])
    assert rc == 0
    benchmark_profile = json.loads(capsys.readouterr().out)
    assert benchmark_profile["workflow_profile"]["profile"] == "benchmark"

    benchmark_run = tmp_path / "benchmark_run"
    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(dataset_root),
            "--suite",
            "industrial-ci",
            "--resize",
            "32",
            "32",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-pretrained",
            "--save-run",
            "--output-dir",
            str(benchmark_run),
            "--suite-export",
            "csv",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    results_jsonl = tmp_path / "results.jsonl"
    rc = infer_main(
        [
            "--model-preset",
            "industrial-template-ncc-map",
            "--train-dir",
            str(dataset_root / "train" / "normal"),
            "--input",
            str(dataset_root / "test"),
            "--save-jsonl",
            str(results_jsonl),
            "--no-pretrained",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert results_jsonl.exists()

    rc = runs_main(["quality", str(benchmark_run), "--json"])
    assert rc == 0
    quality_payload = json.loads(capsys.readouterr().out)
    assert quality_payload["quality"]["status"] in {"reproducible", "audited", "deployable", "limited"}


def test_publish_command_chain_smoke(tmp_path: Path, capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main
    from pyimgano.runs_cli import main as runs_main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    report = export_dir / "report.json"
    config = export_dir / "config.json"
    environment = export_dir / "environment.json"
    leaderboard = export_dir / "leaderboard.csv"
    metadata = export_dir / "leaderboard_metadata.json"

    report.write_text(json.dumps({"suite": "industrial-v4"}), encoding="utf-8")
    config.write_text(json.dumps({"config": {"seed": 123}}), encoding="utf-8")
    environment.write_text(json.dumps({"fingerprint_sha256": "f" * 64}), encoding="utf-8")
    leaderboard.write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": _sha256_file(report),
                    "config_json": _sha256_file(config),
                    "environment_json": _sha256_file(environment),
                },
                "exported_files": {
                    "leaderboard_csv": str(leaderboard),
                    "leaderboard_metadata_json": str(metadata),
                },
                "exported_file_digests": {
                    "leaderboard_csv": _sha256_file(leaderboard),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = doctor_main(["--json", "--profile", "publish", "--publication-target", str(export_dir)])
    assert rc == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["workflow_profile"]["profile"] == "publish"
    assert doctor_payload["publication"]["status"] == "ready"

    rc = runs_main(["acceptance", str(export_dir), "--json"])
    assert rc == 0
    acceptance_payload = json.loads(capsys.readouterr().out)
    assert acceptance_payload["acceptance"]["kind"] == "publication"
    assert acceptance_payload["acceptance"]["ready"] is True

    rc = runs_main(["publication", str(export_dir), "--json"])
    assert rc == 0
    publication_payload = json.loads(capsys.readouterr().out)
    assert publication_payload["publication"]["status"] == "ready"
