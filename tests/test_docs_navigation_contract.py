from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_docs_readme_mentions_start_here_and_benchmark_getting_started() -> None:
    text = _read_text("docs/README_DOCS.md")

    assert "docs/START_HERE.md" in text
    assert "docs/STARTER_PATHS.md" in text
    assert "docs/BENCHMARK_GETTING_STARTED.md" in text
    assert "docs/QUICKSTART.md" in text
    assert "docs/COMPARISON.md" in text
    assert "examples/README.md" in text


def test_algorithm_selection_guide_opens_with_route_oriented_guidance() -> None:
    text = _read_text("docs/ALGORITHM_SELECTION_GUIDE.md")

    assert "If you are starting from the CLI" in text
    assert "pyim --list models --objective" in text
    assert "Your Situation" in text


def test_comparison_doc_mentions_starting_paths() -> None:
    text = _read_text("docs/COMPARISON.md")

    assert "If You Are Deciding Where To Start" in text
    assert "pyimgano-doctor" in text
    assert "pyimgano-demo --smoke" in text


def test_readme_top_navigation_mentions_first_run_and_reference_guides() -> None:
    text = _read_text("README.md")

    assert "docs/START_HERE.md" in text
    assert "docs/STARTER_PATHS.md" in text
    assert "docs/BENCHMARK_GETTING_STARTED.md" in text
    assert "docs/CLI_REFERENCE.md" in text
    assert "docs/ALGORITHM_SELECTION_GUIDE.md" in text


def test_readme_and_start_here_document_guided_workflow() -> None:
    readme = _read_text("README.md")
    start_here = _read_text("docs/START_HERE.md")

    assert "Guided Workflow" in readme
    assert "Discover" in readme
    assert "Benchmark" in readme
    assert "Train" in readme
    assert "Export" in readme
    assert "Infer" in readme
    assert "Validate" in readme
    assert "Gate" in readme
    assert "Guided Workflow" in start_here
    assert "Export" in start_here


def test_starter_paths_doc_maps_goals_to_exact_command_sequences() -> None:
    text = _read_text("docs/STARTER_PATHS.md")

    assert "Deployment Smoke Path" in text
    assert "First 10 Minutes" in text
    assert "Deployment Check" in text
    assert "Publication Gate" in text
    assert "pyimgano --help" in text
    assert "python -m pyimgano --help" in text
    assert "pyimgano-doctor --profile deploy-smoke --json" in text
    assert "pyimgano-doctor --profile first-run --json" in text
    assert "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json" in text
    assert "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json" in text


def test_examples_index_mentions_goal_based_routes() -> None:
    text = _read_text("examples/README.md")

    assert "pyim --goal first-run --json" in text
    assert "pyim --goal pixel-localization --json" in text
    assert "embedding_plus_core_ecod.py" in text
    assert "openclip_plus_core_knn.py" in text
