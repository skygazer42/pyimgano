from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_examples_index_documents_goal_dependency_and_offline_safety() -> None:
    text = _read_text("examples/README.md")

    assert "Goal" in text
    assert "Dependencies" in text
    assert "Offline-safe" in text
    assert "quick_start.py" in text
    assert "industrial_infer_numpy.py" in text

