from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_runs_compare_quality_and_acceptance() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-runs" in text
    assert "compare" in text
    assert "quality" in text
    assert "acceptance" in text


def test_readme_mentions_runs_quality_and_compare_contracts() -> None:
    text = _read_text("README.md")

    assert "pyimgano-runs" in text
    assert "compare" in text
    assert "quality" in text
