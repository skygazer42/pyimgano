from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_run_backed_inference_paths() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--from-run RUN_DIR" in text
    assert "--infer-config PATH" in text
    assert "--profile-json PATH" in text


def test_industrial_inference_documents_stable_runtime_metadata() -> None:
    text = _read_text("docs/INDUSTRIAL_INFERENCE.md")

    assert "decision_summary" in text
    assert "postprocess_summary" in text
    assert "--from-run" in text
    assert "--infer-config" in text


def test_readme_mentions_infer_jsonl_triage_metadata() -> None:
    text = _read_text("README.md")

    assert "decision_summary" in text
    assert "pyimgano-infer" in text
