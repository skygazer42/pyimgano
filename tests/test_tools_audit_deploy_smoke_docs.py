from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_deploy_smoke_docs_reports_missing_requirements(tmp_path: Path) -> None:
    doc = tmp_path / "smoke.md"
    doc.write_text("pyimgano-doctor --profile deploy-smoke --json\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_deploy_smoke_docs.py",
            str(doc),
            "--require",
            "pyimgano-doctor --profile deploy-smoke --json",
            "--require",
            "deploy_smoke_custom_cpu.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "deploy_smoke_custom_cpu.json" in proc.stdout
    assert str(doc) in proc.stdout


def test_audit_deploy_smoke_docs_current_repo_is_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_deploy_smoke_docs.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_audit_deploy_smoke_docs_default_rules_cover_manifest_and_quickstart() -> None:
    import tools.audit_deploy_smoke_docs as audit_deploy_smoke_docs

    paths = {rule.path for rule in audit_deploy_smoke_docs.DEFAULT_RULES}
    assert "docs/QUICKSTART.md" in paths
    assert "docs/MANIFEST_DATASET.md" in paths
