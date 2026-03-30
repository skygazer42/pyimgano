from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_sphinx_index_includes_start_here_and_benchmark_getting_started() -> None:
    text = _read_text("docs/source/index.rst")

    assert "start_here" in text
    assert "benchmark_getting_started" in text


def test_sphinx_start_here_page_links_to_new_onboarding_docs() -> None:
    text = _read_text("docs/source/start_here.rst")

    assert "docs/START_HERE.md" in text
    assert "docs/BENCHMARK_GETTING_STARTED.md" in text
    assert "docs/CLI_REFERENCE.md" in text


def test_sphinx_examples_page_links_to_examples_index() -> None:
    text = _read_text("docs/source/examples.rst")

    assert "examples/README.md" in text


def test_sphinx_benchmarks_page_links_to_starter_benchmark_guide() -> None:
    text = _read_text("docs/source/benchmarks.rst")

    assert "docs/BENCHMARK_GETTING_STARTED.md" in text
