import json


def test_export_suite_tables_writes_metadata_json(tmp_path):
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

    written = export_suite_tables(payload, tmp_path, formats=["csv"])
    metadata_path = tmp_path / "leaderboard_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "leaderboard_csv" in written
    assert metadata["suite"] == "industrial-v4"
    assert metadata["benchmark_config"]["source"].endswith(".json")
    assert metadata["environment_fingerprint_sha256"] == "f" * 64
    assert metadata["split_fingerprint"]["sha256"] == "b" * 64
    assert metadata["citation"]["project"] == "pyimgano"
    assert metadata["citation"]["benchmark_config_source"].endswith(".json")
    assert metadata["citation"]["benchmark_config_sha256"] == "a" * 64
    assert metadata["artifact_quality"]["required_files_present"] is True
    assert metadata["publication_ready"] is True
    assert metadata["evaluation_contract"]["ranking_metric"] == "auroc"
    assert metadata["evaluation_contract"]["metric_directions"]["auroc"] == "higher_is_better"
    assert metadata["evaluation_contract"]["comparability_hints"]["recommends_same_environment"] is True
    assert metadata["exported_files"]["leaderboard_csv"].endswith("leaderboard.csv")
    assert metadata["exported_files"]["leaderboard_metadata_json"].endswith(
        "leaderboard_metadata.json"
    )
