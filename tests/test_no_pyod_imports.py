from __future__ import annotations

import re
from pathlib import Path


_PYOD_IMPORT_RE = re.compile(r"^\\s*(?:from\\s+pyod\\b|import\\s+pyod\\b)", re.MULTILINE)


def _assert_no_pyod_imports(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = _PYOD_IMPORT_RE.search(text)
    assert match is None, f"Found PyOD import in {path}: {match.group(0)!r}"


def test_no_pyod_imports_in_native_bases_and_utils() -> None:
    """
    We are migrating away from PyOD. As an early guardrail, ensure our *native*
    base classes and utility modules do not regress back to importing PyOD.

    Once the migration is complete, we can tighten this to scan the entire
    `pyimgano/` tree.
    """

    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "pyimgano/models/base_detector.py",
        repo_root / "pyimgano/models/base_deep.py",
        repo_root / "pyimgano/models/baseml.py",
        repo_root / "pyimgano/models/baseCv.py",
    ]

    for path in targets:
        assert path.exists(), f"Expected file to exist: {path}"
        _assert_no_pyod_imports(path)

    utils_dir = repo_root / "pyimgano/utils"
    for path in utils_dir.rglob("*.py"):
        _assert_no_pyod_imports(path)

