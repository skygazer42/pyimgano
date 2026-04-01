from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_train_preflight_output_contract() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--preflight" in text
    assert '{"preflight": ...}' in text or '`{"preflight": ...}`' in text
    assert "dataset_readiness" in text
    assert "dataset_issue_codes" in text
