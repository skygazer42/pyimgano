from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_version_files(root: Path, version: str) -> tuple[Path, Path]:
    pyproject = root / "pyproject.toml"
    init_file = root / "__init__.py"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "pyimgano"',
                f'version = "{version}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    init_file.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    return pyproject, init_file


def test_audit_release_version_rejects_prerelease_tag_for_final_version(tmp_path: Path) -> None:
    pyproject, init_file = _write_version_files(tmp_path, "0.9.0")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_release_version.py",
            "--pyproject",
            str(pyproject),
            "--init",
            str(init_file),
            "--tag",
            "v0.9.0-rc1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "release tag 'v0.9.0-rc1' does not match project version '0.9.0'" in proc.stdout


def test_audit_release_version_accepts_hyphenated_rc_tag_for_pep440_rc(
    tmp_path: Path,
) -> None:
    pyproject, init_file = _write_version_files(tmp_path, "0.9.1rc1")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_release_version.py",
            "--pyproject",
            str(pyproject),
            "--init",
            str(init_file),
            "--tag",
            "v0.9.1-rc1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_publish_workflow_checks_release_tag_before_upload() -> None:
    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert 'python tools/audit_release_version.py --tag "$GITHUB_REF_NAME"' in workflow
    assert "if: github.event_name == 'release'" in workflow
