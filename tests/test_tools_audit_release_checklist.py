from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_release_checklist_reports_missing_requirements(tmp_path: Path) -> None:
    doc = tmp_path / "PUBLISHING.md"
    doc.write_text("python -m build\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_release_checklist.py",
            str(doc),
            "--require",
            "python -m build",
            "--require",
            "pyimgano bundle validate",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "pyimgano bundle validate" in proc.stdout
    assert str(doc) in proc.stdout


def test_audit_release_checklist_current_repo_is_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_release_checklist.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
