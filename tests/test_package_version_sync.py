from __future__ import annotations

import re
from pathlib import Path


def test_package_version_matches_pyproject() -> None:
    import pyimgano

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None
    assert pyimgano.__version__ == match.group(1)
