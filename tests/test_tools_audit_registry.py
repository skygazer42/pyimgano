import subprocess
import sys


def test_audit_registry_script_runs():
    subprocess.run([sys.executable, "tools/audit_registry.py"], check=True)

