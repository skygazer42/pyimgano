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
        # As we port models off PyOD, extend this list.
        repo_root / "pyimgano/models/ecod.py",
        repo_root / "pyimgano/models/copod.py",
        repo_root / "pyimgano/models/knn.py",
        repo_root / "pyimgano/models/pca.py",
        repo_root / "pyimgano/models/kde.py",
        repo_root / "pyimgano/models/gmm.py",
        repo_root / "pyimgano/models/iforest.py",
        repo_root / "pyimgano/models/sos.py",
        repo_root / "pyimgano/models/sod.py",
        repo_root / "pyimgano/models/rod.py",
        repo_root / "pyimgano/models/qmcd.py",
        repo_root / "pyimgano/models/lmdd.py",
        repo_root / "pyimgano/models/abod.py",
        repo_root / "pyimgano/models/cof.py",
        repo_root / "pyimgano/models/loci.py",
        repo_root / "pyimgano/models/hbos.py",
        repo_root / "pyimgano/models/mcd.py",
        repo_root / "pyimgano/models/ocsvm.py",
        repo_root / "pyimgano/models/kpca.py",
        repo_root / "pyimgano/models/inne.py",
    ]

    for path in targets:
        assert path.exists(), f"Expected file to exist: {path}"
        _assert_no_pyod_imports(path)

    utils_dir = repo_root / "pyimgano/utils"
    for path in utils_dir.rglob("*.py"):
        _assert_no_pyod_imports(path)
