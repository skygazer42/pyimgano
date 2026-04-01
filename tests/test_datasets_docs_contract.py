from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_dataset_readiness_issue_codes() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-datasets" in text
    assert "readiness_status" in text
    assert "issue_codes" in text
    assert "FEWSHOT_TRAIN_SET" in text
    assert "PIXEL_METRICS_UNAVAILABLE" in text
