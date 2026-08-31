from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_publishing_doc_mentions_release_gate_commands() -> None:
    text = _read_text("docs/PUBLISHING.md")

    assert "python3 tools/audit_deploy_smoke_docs.py" in text
    assert "python3 tools/audit_adoption_docs.py" in text
    assert "python3 tools/audit_release_version.py --tag vX.Y.Z" in text
    assert "python3 tools/audit_release_surface.py" in text
    assert "pyimgano-doctor --profile deploy-smoke --json" in text
    assert "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json" in text
    assert (
        "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json"
        in text
    )
    assert "handoff_report.json" in text


def test_publishing_doc_describes_the_wheel_first_artifact_gate() -> None:
    text = _read_text("docs/PUBLISHING.md")

    assert ".github/workflows/artifact-e2e.yml" in text
    assert "builds exactly one wheel" in text
    assert "downloads the exact wheel that passed" in text
    assert "does not rebuild an untested wheel" in text
    assert "native, ONNX," in text
    assert "TorchScript, and OpenVINO" in text
    assert "documentation command" in text
    assert "artifact security negative tests" in text
    assert "skipped declared runtime" in text
    assert "continue-on-error" in text


def test_publishing_doc_locks_the_release_certified_matrix() -> None:
    text = _read_text("docs/PUBLISHING.md")

    assert "Release-certified artifact matrix" in text
    assert "Ubuntu x86_64 (`ubuntu-latest`)" in text
    assert "| ONNX |" in text and "`CPUExecutionProvider`" in text
    assert "| TorchScript |" in text
    assert "| OpenVINO |" in text
    assert "not release-certified artifact combinations" in text
    assert "runtime-only environments have no importable" in text


def test_publishing_doc_records_schema_trust_and_migration_contract() -> None:
    text = _read_text("docs/PUBLISHING.md")

    assert "ARTIFACT_SCHEMA_VERSION == 1" in text
    assert "trust_checkpoint" in text
    assert "integrity verification still applies" in text
    assert "pyimgano-export --from-run runs/<run_dir>" in text
    assert "deprecated" in text
    assert "embedding-only exporters" in text
    assert "eligible for removal no earlier than `0.11.0`" in text
    assert "`onnx` / `openvino` compatibility extras" in text


def test_stability_doc_publishes_the_same_release_promises() -> None:
    text = _read_text("docs/STABILITY.md")
    normalized = " ".join(text.split())

    assert "`0.10.0`" in text
    assert "schema_version: 1" in text
    assert "ARTIFACT_SCHEMA_VERSION" in text
    assert "Release-certified artifact matrix" in text
    assert "CPUExecutionProvider" in text
    assert "not release-certified" in text
    assert "trust_checkpoint=True" in text
    assert "LegacyArtifactWarning" in text
    assert "eligible for removal no earlier than `0.11.0`" in text
    assert "Package installer aliases cannot emit Python warnings" in normalized
