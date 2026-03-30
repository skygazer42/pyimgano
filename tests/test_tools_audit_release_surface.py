from __future__ import annotations

import subprocess
import sys


def test_audit_release_surface_script_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_release_surface.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout

