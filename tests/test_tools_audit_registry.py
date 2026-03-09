import json
import subprocess
import sys


def test_audit_registry_script_runs():
    subprocess.run([sys.executable, "tools/audit_registry.py"], check=True)


def test_audit_registry_script_can_report_metadata_contract_json():
    proc = subprocess.run(
        [sys.executable, "tools/audit_registry.py", "--metadata-contract", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 1}
    payload = json.loads(proc.stdout)
    assert payload["summary"]["total_models"] > 0
    assert "contract_fields" in payload


def test_audit_registry_metadata_summary_shows_required_issues_but_no_invalid_fields():
    proc = subprocess.run(
        [sys.executable, "tools/audit_registry.py", "--metadata-contract", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 1}
    payload = json.loads(proc.stdout)
    assert payload["summary"]["models_with_invalid_fields"] == 0
