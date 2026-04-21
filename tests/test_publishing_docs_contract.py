from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_publishing_doc_mentions_release_gate_commands() -> None:
    text = _read_text("docs/PUBLISHING.md")

    assert "python3 tools/audit_deploy_smoke_docs.py" in text
    assert "python3 tools/audit_adoption_docs.py" in text
    assert "python3 tools/audit_release_surface.py" in text
    assert "pyimgano-doctor --profile deploy-smoke --json" in text
    assert "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json" in text
    assert (
        "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json"
        in text
    )
    assert "handoff_report.json" in text
