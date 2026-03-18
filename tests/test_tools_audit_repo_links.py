from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_repo_links_reports_legacy_repo_slug(tmp_path: Path) -> None:
    doc = tmp_path / "legacy.md"
    doc.write_text("see https://github.com/jhlu2019/pyimgano/issues\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "tools/audit_repo_links.py", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "jhlu2019/pyimgano" in proc.stdout
    assert str(doc) in proc.stdout


def test_audit_repo_links_reports_placeholder_contact(tmp_path: Path) -> None:
    doc = tmp_path / "contact.md"
    doc.write_text("contact: pyimgano@example.com\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "tools/audit_repo_links.py", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "pyimgano@example.com" in proc.stdout


def test_audit_repo_links_accepts_current_repo_links(tmp_path: Path) -> None:
    doc = tmp_path / "clean.md"
    doc.write_text(
        "\n".join(
            [
                "https://github.com/skygazer42/pyimgano",
                "https://github.com/skygazer42/pyimgano/issues",
                "https://skygazer42.github.io/pyimgano/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "tools/audit_repo_links.py", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "OK" in proc.stdout


def test_audit_repo_links_current_docs_are_clean() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_repo_links.py",
            "README.md",
            "CONTRIBUTING.md",
            "benchmarks/README.md",
            "docs/QUICKSTART.md",
            "docs/DEEP_LEARNING_MODELS.md",
            "docs/source/index.rst",
            "docs/source/contributing.rst",
            "docs/PUBLISHING.md",
            "docs/CAPABILITY_ASSESSMENT.md",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
