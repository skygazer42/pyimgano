from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_bundle_validate_and_run() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-bundle" in text
    assert "validate" in text
    assert "run" in text


def test_readme_mentions_bundle_validate_and_weights_audit() -> None:
    text = _read_text("README.md")

    assert "pyimgano-bundle" in text or "pyimgano bundle" in text
    assert "audit-bundle" in text
