import subprocess
import sys


def test_audit_benchmark_configs_script_runs():
    subprocess.run([sys.executable, "tools/audit_benchmark_configs.py"], check=True)
