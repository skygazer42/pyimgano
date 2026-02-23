import subprocess
import sys


def test_audit_public_api_script_runs():
    subprocess.run([sys.executable, "tools/audit_public_api.py"], check=True)

