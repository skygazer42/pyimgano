from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_demo_smoke_summary_and_next_steps() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-demo" in text
    assert "--smoke" in text
    assert "--scenario" in text
    assert "infer-defects" in text
    assert "--summary-json" in text
    assert "--emit-next-steps" in text


def test_start_here_and_benchmark_getting_started_docs_exist() -> None:
    start_here = _read_text("docs/START_HERE.md")
    starter_paths = _read_text("docs/STARTER_PATHS.md")
    benchmark = _read_text("docs/BENCHMARK_GETTING_STARTED.md")

    assert "pyimgano-doctor" in start_here
    assert "pyimgano-doctor --profile first-run --json" in start_here
    assert "pyimgano-demo --smoke" in start_here
    assert "pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json" in start_here
    assert "pyimgano-benchmark --dataset custom --root ./_demo_custom_dataset" in start_here
    assert "pyimgano runs quality ./_demo_benchmark_run --json" in start_here
    assert "pyimgano-doctor --recommend-extras --for-command benchmark --json" in start_here
    assert "pyimgano-doctor --recommend-extras --for-command train --json" in start_here
    assert "pyimgano-doctor --recommend-extras --for-command infer --json" in start_here
    assert "pyimgano-doctor --recommend-extras --for-command runs --json" in start_here
    assert "--publication-target" in _read_text("docs/CLI_REFERENCE.md")
    assert "--profile publish" in _read_text("docs/CLI_REFERENCE.md")
    assert "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json" in starter_paths
    assert "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json" in starter_paths
    assert "--list-starter-configs" in benchmark
    assert "--starter-config-info" in benchmark
    assert "optional_extras" in benchmark
    assert "optional_baseline_count" in benchmark
    assert "pyimgano-doctor --recommend-extras --for-command benchmark --json" in benchmark
