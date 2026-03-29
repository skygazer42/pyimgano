from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_doctor_extras_and_readiness() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-doctor" in text
    assert "--require-extras" in text
    assert "--accelerators" in text
    assert "--run-dir" in text
    assert "--deploy-bundle" in text


def test_readme_mentions_doctor_require_extras_and_readiness() -> None:
    text = _read_text("README.md")

    assert "pyimgano-doctor" in text
    assert "--suite industrial-v4" in text
