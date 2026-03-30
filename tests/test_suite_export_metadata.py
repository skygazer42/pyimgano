import hashlib
import json


def _write_run_artifacts(root):
    (root / "report.json").write_text(json.dumps({"suite": "industrial-v4"}), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"config": {"seed": 123}}), encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_suite_tables_writes_metadata_json(tmp_path):
    from pyimgano.reporting.suite_export import export_suite_tables

    _write_run_artifacts(tmp_path)
    payload = {
        "suite": "industrial-v4",
        "dataset": "mvtec",
        "category": "bottle",
        "dataset_profile": {
            "total_records": 15,
            "train_count": 10,
            "test_count": 5,
            "test_normal_count": 2,
            "test_anomaly_count": 3,
            "categories": ["bottle"],
            "category_count": 1,
            "has_masks": True,
            "pixel_metrics_available": True,
            "fewshot_risk": False,
            "multi_category": False,
        },
        "rows": [
            {
                "name": "a",
                "model": "vision_patchcore_inspection_checkpoint",
                "auroc": 0.95,
                "run_dir": "runs/a",
                "deployment_profile": {
                    "family": ["patchcore", "memory_bank"],
                    "training_regime": "checkpoint-wrapper",
                    "runtime_cost_hint": "high",
                    "memory_cost_hint": "high",
                    "artifact_requirements": ["checkpoint"],
                    "upstream_project": "patchcore_inspection",
                    "industrial_fit": {"pixel_localization": True},
                },
            }
        ],
        "split_fingerprint": {
            "schema_version": 1,
            "sha256": "b" * 64,
            "train_count": 10,
            "calibration_count": 0,
            "test_count": 5,
        },
        "benchmark_config": {
            "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
            "official": True,
            "sha256": "a" * 64,
        },
        "environment_fingerprint_sha256": "f" * 64,
    }

    written = export_suite_tables(payload, tmp_path, formats=["csv"])
    metadata_path = tmp_path / "leaderboard_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "leaderboard_csv" in written
    assert metadata["suite"] == "industrial-v4"
    assert metadata["benchmark_config"]["source"].endswith(".json")
    assert metadata["benchmark_config"]["starter"] is True
    assert metadata["benchmark_config"]["starter_tier"] == "starter"
    assert metadata["benchmark_config"]["optional_extras"] == ["clip", "skimage", "torch"]
    assert metadata["benchmark_config"]["optional_baseline_count"] == 11
    assert metadata["benchmark_config"]["starter_list_command"] == "pyimgano benchmark --list-starter-configs"
    assert metadata["environment_fingerprint_sha256"] == "f" * 64
    assert metadata["split_fingerprint"]["sha256"] == "b" * 64
    assert metadata["dataset_profile"]["pixel_metrics_available"] is True
    assert metadata["deployment_summary"]["model_count"] == 1
    assert metadata["deployment_summary"]["families"] == ["memory_bank", "patchcore"]
    assert metadata["deployment_summary"]["runtime_cost_hints"] == {"high": 1}
    assert metadata["deployment_summary"]["memory_cost_hints"] == {"high": 1}
    assert metadata["deployment_summary"]["artifact_requirements"] == ["checkpoint"]
    assert metadata["deployment_summary"]["pixel_localization_models"] == 1
    assert metadata["upstream_coverage_summary"]["native"] == 0
    assert metadata["upstream_coverage_summary"]["anomalib"] == 0
    assert metadata["upstream_coverage_summary"]["patchcore_inspection"] == 1
    assert metadata["benchmark_context"]["pixel_metrics_available"] is True
    assert metadata["benchmark_context"]["multi_category"] is False
    assert "checkpoint_models_require_artifact_governance" in metadata["constraint_warnings"]
    assert metadata["citation"]["project"] == "pyimgano"
    assert metadata["citation"]["benchmark_config_source"].endswith(".json")
    assert metadata["citation"]["benchmark_config_sha256"] == "a" * 64
    assert metadata["artifact_quality"]["required_files_present"] is True
    assert metadata["artifact_quality"]["has_evaluation_contract"] is True
    assert metadata["artifact_quality"]["has_benchmark_citation"] is True
    assert metadata["artifact_quality"]["has_benchmark_provenance"] is True
    assert metadata["artifact_quality"]["has_run_artifact_refs"] is True
    assert metadata["artifact_quality"]["has_run_artifact_digests"] is True
    assert metadata["artifact_quality"]["has_exported_file_digests"] is True
    assert metadata["publication_ready"] is True
    assert metadata["evaluation_contract"]["ranking_metric"] == "auroc"
    assert metadata["evaluation_contract"]["metric_directions"]["auroc"] == "higher_is_better"
    assert (
        metadata["evaluation_contract"]["comparability_hints"]["recommends_same_environment"]
        is True
    )
    assert metadata["exported_files"]["leaderboard_csv"].endswith("leaderboard.csv")
    assert metadata["exported_files"]["best_by_baseline_csv"].endswith("best_by_baseline.csv")
    assert metadata["exported_files"]["skipped_csv"].endswith("skipped.csv")
    assert metadata["exported_files"]["leaderboard_csv"] == "leaderboard.csv"
    assert metadata["exported_files"]["best_by_baseline_csv"] == "best_by_baseline.csv"
    assert metadata["exported_files"]["skipped_csv"] == "skipped.csv"
    assert metadata["exported_files"]["leaderboard_metadata_json"] == "leaderboard_metadata.json"
    assert metadata["exported_file_digests"]["leaderboard_csv"] == _sha256(
        tmp_path / "leaderboard.csv"
    )
    assert metadata["exported_file_digests"]["best_by_baseline_csv"] == _sha256(
        tmp_path / "best_by_baseline.csv"
    )
    assert metadata["exported_file_digests"]["skipped_csv"] == _sha256(tmp_path / "skipped.csv")
    assert metadata["audit_refs"]["report_json"] == "report.json"
    assert metadata["audit_refs"]["config_json"] == "config.json"
    assert metadata["audit_refs"]["environment_json"] == "environment.json"
    assert metadata["audit_digests"]["report_json"] == _sha256(tmp_path / "report.json")
    assert metadata["audit_digests"]["config_json"] == _sha256(tmp_path / "config.json")
    assert metadata["audit_digests"]["environment_json"] == _sha256(tmp_path / "environment.json")


def test_export_suite_tables_requires_benchmark_provenance_for_publication_ready(tmp_path):
    from pyimgano.reporting.suite_export import export_suite_tables

    _write_run_artifacts(tmp_path)
    payload = {
        "suite": "industrial-v4",
        "dataset": "mvtec",
        "category": "bottle",
        "rows": [{"name": "a", "auroc": 0.95, "run_dir": "runs/a"}],
        "split_fingerprint": {
            "schema_version": 1,
            "sha256": "b" * 64,
            "train_count": 10,
            "calibration_count": 0,
            "test_count": 5,
        },
        "benchmark_config": {
            "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
            "official": True,
        },
        "environment_fingerprint_sha256": "f" * 64,
    }

    export_suite_tables(payload, tmp_path, formats=["csv"])
    metadata = json.loads((tmp_path / "leaderboard_metadata.json").read_text(encoding="utf-8"))

    assert metadata["artifact_quality"]["has_benchmark_provenance"] is False
    assert "benchmark_config.sha256" in metadata["artifact_quality"]["missing_required"]
    assert metadata["publication_ready"] is False


def test_export_suite_tables_requires_run_artifact_refs_for_publication_ready(tmp_path):
    from pyimgano.reporting.suite_export import export_suite_tables

    payload = {
        "suite": "industrial-v4",
        "dataset": "mvtec",
        "category": "bottle",
        "rows": [{"name": "a", "auroc": 0.95, "run_dir": "runs/a"}],
        "split_fingerprint": {
            "schema_version": 1,
            "sha256": "b" * 64,
            "train_count": 10,
            "calibration_count": 0,
            "test_count": 5,
        },
        "benchmark_config": {
            "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
            "official": True,
            "sha256": "a" * 64,
        },
        "environment_fingerprint_sha256": "f" * 64,
    }

    export_suite_tables(payload, tmp_path, formats=["csv"])
    metadata = json.loads((tmp_path / "leaderboard_metadata.json").read_text(encoding="utf-8"))

    assert metadata["artifact_quality"]["has_run_artifact_refs"] is False
    assert "audit_refs.report_json" in metadata["artifact_quality"]["missing_required"]
    assert "audit_refs.config_json" in metadata["artifact_quality"]["missing_required"]
    assert "audit_refs.environment_json" in metadata["artifact_quality"]["missing_required"]
    assert metadata["artifact_quality"]["has_run_artifact_digests"] is False
    assert "audit_digests.report_json" in metadata["artifact_quality"]["missing_required"]
    assert "audit_digests.config_json" in metadata["artifact_quality"]["missing_required"]
    assert "audit_digests.environment_json" in metadata["artifact_quality"]["missing_required"]
    assert metadata["artifact_quality"]["has_exported_file_digests"] is True
    assert metadata["publication_ready"] is False


def test_export_suite_tables_writes_category_matrix_files(tmp_path):
    from pyimgano.reporting.suite_export import export_suite_tables

    _write_run_artifacts(tmp_path)
    payload = {
        "suite": "industrial-v4",
        "dataset": "mvtec",
        "category": "all",
        "rows": [
            {
                "name": "baseline_a",
                "base_name": "baseline_a",
                "variant": "base",
                "auroc": 0.8,
                "average_precision": 0.7,
                "run_dir": "runs/a",
            }
        ],
        "matrix": {
            "scope": "per_category",
            "categories": ["bottle", "capsule"],
            "metrics": ["auroc", "average_precision"],
            "by_metric": {
                "auroc": [
                    {
                        "name": "baseline_a",
                        "base_name": "baseline_a",
                        "variant": "base",
                        "mean": 0.8,
                        "std": 0.1,
                        "values": {"bottle": 0.9, "capsule": 0.7},
                    }
                ],
                "average_precision": [
                    {
                        "name": "baseline_a",
                        "base_name": "baseline_a",
                        "variant": "base",
                        "mean": 0.7,
                        "std": 0.05,
                        "values": {"bottle": 0.75, "capsule": 0.65},
                    }
                ],
            },
        },
        "split_fingerprint": {
            "schema_version": 1,
            "sha256": "b" * 64,
            "train_count": 10,
            "calibration_count": 0,
            "test_count": 5,
        },
        "benchmark_config": {
            "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
            "official": True,
            "sha256": "a" * 64,
        },
        "environment_fingerprint_sha256": "f" * 64,
    }

    written = export_suite_tables(payload, tmp_path, formats=["csv"])
    metadata = json.loads((tmp_path / "leaderboard_metadata.json").read_text(encoding="utf-8"))

    assert written["category_matrix_auroc_csv"].endswith("category_matrix_auroc.csv")
    assert written["category_matrix_average_precision_csv"].endswith(
        "category_matrix_average_precision.csv"
    )
    assert (tmp_path / "category_matrix_auroc.csv").exists()
    assert (tmp_path / "category_matrix_average_precision.csv").exists()
    matrix_csv = (tmp_path / "category_matrix_auroc.csv").read_text(encoding="utf-8")
    assert "bottle" in matrix_csv
    assert "capsule" in matrix_csv
    assert "baseline_a" in matrix_csv
    assert metadata["exported_files"]["category_matrix_auroc_csv"].endswith(
        "category_matrix_auroc.csv"
    )
    assert metadata["exported_files"]["category_matrix_average_precision_csv"].endswith(
        "category_matrix_average_precision.csv"
    )
    assert metadata["exported_file_digests"]["category_matrix_auroc_csv"] == _sha256(
        tmp_path / "category_matrix_auroc.csv"
    )
