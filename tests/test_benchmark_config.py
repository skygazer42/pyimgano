import json


def test_describe_benchmark_config_marks_official_config(tmp_path):
    from pyimgano.reporting.benchmark_config import describe_benchmark_config

    cfg = tmp_path / "official_mvtec_industrial_v4_cpu_offline.json"
    cfg.write_text(json.dumps({"dataset": "mvtec", "suite": "industrial-v4"}), encoding="utf-8")

    info = describe_benchmark_config(str(cfg))

    assert info["source"] == str(cfg)
    assert info["official"] is True
    assert info["kind"] == "file"


def test_validate_benchmark_config_payload_requires_dataset_or_suite():
    from pyimgano.reporting.benchmark_config import validate_benchmark_config_payload

    errors = validate_benchmark_config_payload({"root": "/tmp/data"})

    assert any("dataset" in item for item in errors)


def test_list_official_benchmark_configs_returns_structured_metadata():
    from pyimgano.reporting.benchmark_config import list_official_benchmark_configs

    payload = list_official_benchmark_configs()

    assert isinstance(payload, list)
    by_name = {str(item["name"]): item for item in payload}
    mvtec = by_name["official_mvtec_industrial_v4_cpu_offline.json"]
    assert mvtec["official"] is True
    assert mvtec["dataset"] == "mvtec"
    assert mvtec["suite"] == "industrial-v4"
    assert mvtec["errors"] == []
    assert isinstance(mvtec["sha256"], str)
    assert mvtec["sha256"]


def test_describe_benchmark_config_can_resolve_official_name_without_full_path():
    from pyimgano.reporting.benchmark_config import describe_benchmark_config

    info = describe_benchmark_config("official_mvtec_industrial_v4_cpu_offline.json")

    assert info["name"] == "official_mvtec_industrial_v4_cpu_offline.json"
    assert info["official"] is True
    assert info["dataset"] == "mvtec"
    assert info["suite"] == "industrial-v4"
    assert info["source"].endswith("official_mvtec_industrial_v4_cpu_offline.json")
    assert info["evaluation_contract"]["ranking_metric"] == "auroc"
    assert info["evaluation_contract"]["metric_directions"]["pixel_auroc"] == "higher_is_better"
    assert info["evaluation_contract"]["comparability_hints"]["requires_same_split"] is True
    trust = info["trust_summary"]
    assert trust["status"] == "trust-signaled"
    assert trust["trust_signals"]["is_official"] is True
    assert trust["trust_signals"]["has_source_path"] is True
    assert trust["trust_signals"]["has_sha256"] is True
    assert trust["trust_signals"]["has_dataset"] is True
    assert trust["trust_signals"]["has_suite_or_model"] is True
    assert trust["trust_signals"]["has_evaluation_contract"] is True
    assert trust["audit_refs"]["benchmark_config_source"].endswith(
        "official_mvtec_industrial_v4_cpu_offline.json"
    )
    assert info["starter"] is True
    assert info["starter_tier"] == "starter"
    assert info["optional_extras"] == ["clip", "skimage", "torch"]
    assert info["optional_baseline_count"] == 11
    assert info["starter_list_command"] == "pyimgano benchmark --list-starter-configs"
    assert (
        info["starter_info_command"]
        == "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json"
    )
    assert (
        info["starter_run_command"]
        == "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json"
    )
