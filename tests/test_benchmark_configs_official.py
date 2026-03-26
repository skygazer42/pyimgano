from pathlib import Path


def test_official_benchmark_configs_validate():
    from pyimgano.reporting.benchmark_config import load_and_validate_benchmark_config

    cfg_dir = Path("benchmarks/configs")
    paths = sorted(cfg_dir.glob("official_*.json"))
    assert paths
    for path in paths:
        payload = load_and_validate_benchmark_config(path)
        assert "dataset" in payload or "suite" in payload
