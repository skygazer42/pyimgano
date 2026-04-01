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
    assert "dataset_readiness" in text
    assert "dataset_issue_codes" in text


def test_compare_docs_document_dataset_readiness_summary_contract() -> None:
    cli_text = _read_text("docs/CLI_REFERENCE.md")
    run_compare_text = _read_text("docs/RUN_COMPARISON.md")

    for text in (cli_text, run_compare_text):
        assert "baseline_dataset_readiness" in text
        assert "candidate_dataset_readiness" in text
        assert "dataset_readiness_status" in text


def test_readme_mentions_runs_quality_and_compare_contracts() -> None:
    text = _read_text("README.md")

    assert "pyimgano-runs" in text
    assert "compare" in text
    assert "quality" in text
    assert "dataset_readiness" in text
