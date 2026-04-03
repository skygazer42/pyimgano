from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_bundle_validate_run_and_watch() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-bundle" in text
    assert "validate" in text
    assert "run" in text
    assert "watch" in text
    assert "webhook" in text
    assert "webhook-header" in text
    assert "webhook-bearer-token" in text
    assert "webhook-signing-secret" in text
    assert "webhook-bearer-token-env" in text
    assert "webhook-signing-secret-env" in text
    assert "bundle_manifest.json" in text
    assert "handoff_report_status" in text
    assert "next_action" in text
    assert "watch_state.json" in text
    assert "watch_events.jsonl" in text


def test_readme_mentions_bundle_validate_watch_and_weights_audit() -> None:
    text = _read_text("README.md")

    assert "pyimgano-bundle" in text or "pyimgano bundle" in text
    assert "bundle watch" in text
    assert "webhook" in text
    assert "webhook-header" in text
    assert "webhook-signing-secret" in text
    assert "webhook-bearer-token-env" in text
    assert "audit-bundle" in text
