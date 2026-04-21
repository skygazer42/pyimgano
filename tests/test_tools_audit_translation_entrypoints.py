from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_translation_entrypoints_reports_missing_command(tmp_path: Path) -> None:
    doc = tmp_path / "README_cn.md"
    doc.write_text("# pyimgano\npyimgano-train\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "tools/audit_translation_entrypoints.py", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "missing translation entrypoint" in proc.stdout
    assert "pyimgano-demo --smoke" in proc.stdout


def test_audit_translation_entrypoints_current_repo_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_translation_entrypoints.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
