from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_artifacts(export_dir: Path) -> None:
    _write_json(export_dir / "report.json", {"suite": "industrial-v4"})
    _write_json(export_dir / "config.json", {"config": {"seed": 123}})
    _write_json(export_dir / "environment.json", {"fingerprint_sha256": "f" * 64})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_map(**paths: Path) -> dict[str, str]:
    return {key: _sha256(path) for key, path in paths.items()}


def test_evaluate_publication_quality_detects_ready_export(tmp_path: Path) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(export_dir / "leaderboard_metadata.json", {
        "artifact_quality": {
            "required_files_present": True,
            "missing_required": [],
            "has_official_benchmark_config": True,
            "has_environment_fingerprint": True,
            "has_split_fingerprint": True,
        },
        "benchmark_config": {
            "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
            "official": True,
            "sha256": "a" * 64,
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
            "report_json": _sha256(export_dir / "report.json"),
            "config_json": _sha256(export_dir / "config.json"),
            "environment_json": _sha256(export_dir / "environment.json"),
        },
        "exported_files": {
            "leaderboard_csv": str(export_dir / "leaderboard.csv"),
            "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
        },
            "exported_file_digests": {
                "leaderboard_csv": _sha256(export_dir / "leaderboard.csv"),
            },
    })

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "ready"
    assert quality["publication_ready"] is True
    assert quality["missing_required"] == []
    assert quality["exported_files_present"]["leaderboard_csv"] is True
    assert quality["trust_signals"] == {
        "has_official_benchmark_config": True,
        "has_evaluation_contract": True,
        "has_benchmark_citation": True,
        "has_cross_checked_assets": False,
        "has_benchmark_provenance": True,
        "has_benchmark_config_ref": True,
        "has_run_artifact_refs": True,
        "has_run_artifact_digests": True,
        "has_exported_file_digests": True,
    }
    assert quality["audit_refs"]["leaderboard_metadata_json"] == "leaderboard_metadata.json"
    assert quality["audit_refs"]["leaderboard_csv"] == "leaderboard.csv"
    assert quality["audit_refs"]["benchmark_config_source"].endswith(".json")
    assert quality["audit_refs"]["report_json"] == "report.json"
    assert quality["audit_refs"]["config_json"] == "config.json"
    assert quality["audit_refs"]["environment_json"] == "environment.json"
    assert quality["audit_digests"]["report_json"] == _sha256(export_dir / "report.json")
    assert quality["audit_digests"]["config_json"] == _sha256(export_dir / "config.json")
    assert quality["audit_digests"]["environment_json"] == _sha256(export_dir / "environment.json")
    assert quality["exported_file_digests"]["leaderboard_csv"] == _sha256(
        export_dir / "leaderboard.csv"
    )


def test_evaluate_publication_quality_reports_partial_export(tmp_path: Path) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_json(export_dir / "leaderboard_metadata.json", {
        "artifact_quality": {
            "required_files_present": False,
            "missing_required": ["environment_fingerprint_sha256"],
            "has_official_benchmark_config": False,
            "has_environment_fingerprint": False,
            "has_split_fingerprint": True,
        },
        "publication_ready": False,
        "exported_files": {
            "leaderboard_csv": str(export_dir / "leaderboard.csv"),
            "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
        },
    })

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "partial"
    assert quality["publication_ready"] is False
    assert quality["missing_required"] == [
        "environment_fingerprint_sha256",
        "leaderboard_csv",
        "exported_file_digests.leaderboard_csv",
        "audit_refs.report_json",
        "audit_refs.config_json",
        "audit_refs.environment_json",
        "audit_digests.report_json",
        "audit_digests.config_json",
        "audit_digests.environment_json",
        "benchmark_config",
        "split_fingerprint.sha256",
        "evaluation_contract",
        "citation",
    ]
    assert quality["exported_files_present"]["leaderboard_csv"] is False
    assert quality["trust_signals"]["has_benchmark_provenance"] is False
    assert quality["trust_signals"]["has_exported_file_digests"] is False


def test_evaluate_publication_quality_blocks_missing_citation_even_when_flagged_ready(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(
        export_dir / "leaderboard_metadata.json",
        {
            "artifact_quality": {
                "required_files_present": True,
                "missing_required": [],
                "has_official_benchmark_config": True,
                "has_environment_fingerprint": True,
                "has_split_fingerprint": True,
            },
            "benchmark_config": {
                "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                "official": True,
                "sha256": "a" * 64,
            },
            "environment_fingerprint_sha256": "f" * 64,
            "split_fingerprint": {"sha256": "b" * 64},
            "evaluation_contract": {"primary_metric": "auroc"},
            "publication_ready": True,
            "audit_refs": {
                "report_json": "report.json",
                "config_json": "config.json",
                "environment_json": "environment.json",
            },
            "audit_digests": {
                "report_json": _sha256(export_dir / "report.json"),
                "config_json": _sha256(export_dir / "config.json"),
                "environment_json": _sha256(export_dir / "environment.json"),
            },
            "exported_files": {
                "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
            },
            "exported_file_digests": {
                "leaderboard_csv": _sha256(export_dir / "leaderboard.csv"),
            },
        },
    )

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "partial"
    assert quality["publication_ready"] is False
    assert "citation" in quality["missing_required"]
    assert quality["trust_signals"]["has_benchmark_citation"] is False


def test_evaluate_publication_quality_blocks_missing_run_artifact_refs_even_when_files_exist(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(
        export_dir / "leaderboard_metadata.json",
        {
            "artifact_quality": {
                "required_files_present": True,
                "missing_required": [],
                "has_official_benchmark_config": True,
                "has_environment_fingerprint": True,
                "has_split_fingerprint": True,
            },
            "benchmark_config": {
                "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                "official": True,
                "sha256": "a" * 64,
            },
            "environment_fingerprint_sha256": "f" * 64,
            "split_fingerprint": {"sha256": "b" * 64},
            "evaluation_contract": {"primary_metric": "auroc"},
            "citation": {"project": "pyimgano"},
            "publication_ready": True,
            "exported_files": {
                "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
            },
            "exported_file_digests": {
                "leaderboard_csv": _sha256(export_dir / "leaderboard.csv"),
            },
        },
    )

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "partial"
    assert quality["publication_ready"] is False
    assert "audit_refs.report_json" in quality["missing_required"]
    assert "audit_refs.config_json" in quality["missing_required"]
    assert "audit_refs.environment_json" in quality["missing_required"]
    assert quality["trust_signals"]["has_run_artifact_refs"] is False


def test_evaluate_publication_quality_rejects_run_artifact_refs_outside_export_root(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.publication_quality import _resolve_exported_path

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    assert _resolve_exported_path(export_dir, "../outside/report.json") is None
    assert _resolve_exported_path(export_dir, "../../escape.json") is None


def test_evaluate_publication_quality_blocks_mismatched_run_artifact_digest(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(
        export_dir / "leaderboard_metadata.json",
        {
            "artifact_quality": {
                "required_files_present": True,
                "missing_required": [],
                "has_official_benchmark_config": True,
                "has_environment_fingerprint": True,
                "has_split_fingerprint": True,
            },
            "benchmark_config": {
                "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                "official": True,
                "sha256": "a" * 64,
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
                "report_json": "0" * 64,
                "config_json": _sha256(export_dir / "config.json"),
                "environment_json": _sha256(export_dir / "environment.json"),
            },
            "exported_files": {
                "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
            },
            "exported_file_digests": {
                "leaderboard_csv": _sha256(export_dir / "leaderboard.csv"),
            },
        },
    )

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "partial"
    assert quality["publication_ready"] is False
    assert "report_json_sha256_mismatch" in quality["missing_required"]
    assert quality["trust_signals"]["has_run_artifact_digests"] is False


def test_evaluate_publication_quality_blocks_mismatched_exported_file_digest(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(
        export_dir / "leaderboard_metadata.json",
        {
            "artifact_quality": {
                "required_files_present": True,
                "missing_required": [],
                "has_official_benchmark_config": True,
                "has_environment_fingerprint": True,
                "has_split_fingerprint": True,
            },
            "benchmark_config": {
                "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                "official": True,
                "sha256": "a" * 64,
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
                "report_json": _sha256(export_dir / "report.json"),
                "config_json": _sha256(export_dir / "config.json"),
                "environment_json": _sha256(export_dir / "environment.json"),
            },
            "exported_files": {
                "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
            },
            "exported_file_digests": {
                "leaderboard_csv": "0" * 64,
            },
        },
    )

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "partial"
    assert quality["publication_ready"] is False
    assert "leaderboard_csv_sha256_mismatch" in quality["missing_required"]
    assert quality["trust_signals"]["has_exported_file_digests"] is False


def test_evaluate_publication_quality_accepts_valid_declared_weight_artifacts(tmp_path: Path) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    model_card_path = export_dir / "model_card.json"
    manifest_path = export_dir / "weights_manifest.json"
    _write_json(
        model_card_path,
        {
            "schema_version": 1,
            "model_name": "demo_model",
            "summary": {
                "purpose": "demo",
                "intended_inputs": "RGB",
                "output_contract": "image-level",
            },
            "weights": {
                "path": "checkpoints/demo.pt",
                "manifest_entry": "demo_model_entry",
                "source": "unit-test",
                "license": "internal",
            },
            "deployment": {"runtime": "torch"},
        },
    )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "entries": [
                {
                    "name": "demo_model_entry",
                    "path": "checkpoints/demo.pt",
                    "source": "unit-test",
                    "license": "internal",
                    "runtime": "torch",
                }
            ],
        },
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(
        export_dir / "leaderboard_metadata.json",
        {
            "artifact_quality": {
                "required_files_present": True,
                "missing_required": [],
                "has_official_benchmark_config": True,
                "has_environment_fingerprint": True,
                "has_split_fingerprint": True,
            },
            "benchmark_config": {
                "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                "official": True,
                "sha256": "a" * 64,
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
                "report_json": _sha256(export_dir / "report.json"),
                "config_json": _sha256(export_dir / "config.json"),
                "environment_json": _sha256(export_dir / "environment.json"),
            },
            "exported_files": {
                "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                "model_card_json": str(model_card_path),
                "weights_manifest_json": str(manifest_path),
            },
            "exported_file_digests": _sha256_map(
                leaderboard_csv=export_dir / "leaderboard.csv",
                model_card_json=model_card_path,
                weights_manifest_json=manifest_path,
            ),
        },
    )

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "ready"
    assert quality["publication_ready"] is True
    assert quality["invalid_declared"] == []
    assert quality["asset_audit"]["valid"] is True
    assert quality["asset_audit"]["model_card"]["valid"] is True
    assert quality["asset_audit"]["weights_manifest"]["valid"] is True
    assert quality["trust_signals"]["has_cross_checked_assets"] is True


def test_evaluate_publication_quality_reports_invalid_declared_model_card(tmp_path: Path) -> None:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    export_dir = tmp_path / "suite_export"
    _write_run_artifacts(export_dir)
    model_card_path = export_dir / "model_card.json"
    _write_json(
        model_card_path,
        {
            "schema_version": 1,
            "model_name": "broken_model",
            "weights": {
                "path": "checkpoints/demo.pt",
                "source": "unit-test",
                "license": "internal",
            },
            "deployment": {"runtime": "torch"},
        },
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    _write_json(
        export_dir / "leaderboard_metadata.json",
        {
            "artifact_quality": {
                "required_files_present": True,
                "missing_required": [],
                "has_official_benchmark_config": True,
                "has_environment_fingerprint": True,
                "has_split_fingerprint": True,
            },
            "benchmark_config": {
                "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                "official": True,
                "sha256": "a" * 64,
            },
            "environment_fingerprint_sha256": "f" * 64,
            "split_fingerprint": {"sha256": "b" * 64},
            "citation": {"project": "pyimgano"},
            "evaluation_contract": {"primary_metric": "auroc"},
            "publication_ready": True,
            "audit_refs": {
                "report_json": "report.json",
                "config_json": "config.json",
                "environment_json": "environment.json",
            },
            "audit_digests": {
                "report_json": _sha256(export_dir / "report.json"),
                "config_json": _sha256(export_dir / "config.json"),
                "environment_json": _sha256(export_dir / "environment.json"),
            },
            "exported_files": {
                "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                "model_card_json": str(model_card_path),
            },
            "exported_file_digests": _sha256_map(
                leaderboard_csv=export_dir / "leaderboard.csv",
                model_card_json=model_card_path,
            ),
        },
    )

    quality = evaluate_publication_quality(export_dir)

    assert quality["status"] == "partial"
    assert quality["publication_ready"] is False
    assert quality["invalid_declared"] == ["model_card_json"]
    assert quality["asset_audit"]["valid"] is False
    assert quality["asset_audit"]["model_card"]["valid"] is False
