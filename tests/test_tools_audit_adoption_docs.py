from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_adoption_docs_reports_missing_required_needles(tmp_path: Path) -> None:
    doc = tmp_path / "README.md"
    doc.write_text("# title\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "tools/audit_adoption_docs.py", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "missing required adoption doc entrypoint" in proc.stdout
    assert str(doc) in proc.stdout


def test_audit_adoption_docs_current_repo_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_adoption_docs.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
