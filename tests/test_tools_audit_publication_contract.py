from __future__ import annotations

import subprocess
import sys


def test_audit_publication_contract_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "tools/audit_publication_contract.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "trust-signaled" in result.stdout.lower()
