from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_pyim_starter_pick_metadata() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--goal" in text
    assert "--objective" in text
    assert "--selection-profile" in text
    assert "--topk" in text
    assert "supports_pixel_map" in text
    assert "tested_runtime" in text
    assert "why_this_pick" in text
    assert "install hint" in text
    assert "Selection Context" in text
    assert "Goal Context" in text
    assert "Goal Picks" in text
    assert "Suggested Commands" in text


def test_algorithm_selection_guide_and_examples_document_pyim_goals() -> None:
    guide = _read_text("docs/ALGORITHM_SELECTION_GUIDE.md")
    examples = _read_text("examples/README.md")

    assert "pyim --goal first-run --json" in guide
    assert "pyim --goal deployable --json" in guide
    assert "pyim --goal first-run --json" in examples
    assert "baseline" in examples.lower()
    assert "optional backend" in examples.lower()
