from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_docs_mention_python_module_entrypoint() -> None:
    readme = _read_text("README.md")
    cli_reference = _read_text("docs/CLI_REFERENCE.md")
    start_here = _read_text("docs/START_HERE.md")
    starter_paths = _read_text("docs/STARTER_PATHS.md")

    assert "python -m pyimgano --help" in readme
    assert "python -m pyimgano --help" in cli_reference
    assert "python -m pyimgano --help" in start_here
    assert "python -m pyimgano --help" in starter_paths
    assert "guided workflow" in cli_reference.lower()
