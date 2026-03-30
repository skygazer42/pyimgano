from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_pyim_starter_pick_metadata() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--objective" in text
    assert "--selection-profile" in text
    assert "--topk" in text
    assert "supports_pixel_map" in text
    assert "tested_runtime" in text
    assert "install hint" in text
    assert "Selection Context" in text
    assert "Suggested Commands" in text
