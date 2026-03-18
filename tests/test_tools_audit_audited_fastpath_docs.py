from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_audited_fastpath_docs_reports_missing_requirements(tmp_path: Path) -> None:
    doc = tmp_path / "fastpath.md"
    doc.write_text("infer_config.json\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_audited_fastpath_docs.py",
            str(doc),
            "--require",
            "infer_config.json",
            "--require",
            "bundle_manifest.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "bundle_manifest.json" in proc.stdout
    assert str(doc) in proc.stdout


def test_audit_audited_fastpath_docs_accepts_complete_doc(tmp_path: Path) -> None:
    doc = tmp_path / "fastpath.md"
    doc.write_text(
        "\n".join(
            [
                "infer_config.json",
                "calibration_card.json",
                "bundle_manifest.json",
                "industrial_adapt_audited.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_audited_fastpath_docs.py",
            str(doc),
            "--require",
            "infer_config.json",
            "--require",
            "calibration_card.json",
            "--require",
            "bundle_manifest.json",
            "--require",
            "industrial_adapt_audited.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "OK" in proc.stdout


def test_audit_audited_fastpath_docs_current_repo_is_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_audited_fastpath_docs.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
