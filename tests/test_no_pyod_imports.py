from __future__ import annotations

import re
from pathlib import Path


_PYOD_IMPORT_RE = re.compile(r"^\s*(?:from\s+pyod\b|import\s+pyod\b)", re.MULTILINE)


def _assert_no_pyod_imports(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = _PYOD_IMPORT_RE.search(text)
    assert match is None, f"Found PyOD import in {path}: {match.group(0)!r}"


def test_no_pyod_imports_in_native_bases_and_utils() -> None:
    """
    Guardrail: `pyimgano` must not import PyOD at runtime.

    We intentionally scan the whole package tree (not docs/tests) so that new
    detectors cannot quietly reintroduce `pyod` as a hard dependency.
    """

    repo_root = Path(__file__).resolve().parents[1]
    package_dir = repo_root / "pyimgano"
    for path in package_dir.rglob("*.py"):
        _assert_no_pyod_imports(path)
