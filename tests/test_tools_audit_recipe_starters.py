from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_recipe_starters_reports_recipe_mismatch(tmp_path: Path) -> None:
    cfg = tmp_path / "starter.json"
    cfg.write_text('{"recipe":"wrong-recipe"}', encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_recipe_starters.py",
            "--recipe-name",
            "expected-recipe",
            "--config-path",
            str(cfg),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "expected-recipe" in proc.stdout
    assert "wrong-recipe" in proc.stdout
    assert str(cfg) in proc.stdout


def test_audit_recipe_starters_current_repo_is_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_recipe_starters.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_audit_recipe_starters_requires_explicit_status_when_no_checked_in_config(
    tmp_path: Path,
) -> None:
    import tools.audit_recipe_starters as audit_recipe_starters

    issues = audit_recipe_starters._validate_recipe_metadata(
        recipe_name="recipe-without-starter",
        metadata={},
        repo_root=tmp_path,
    )

    assert any("starter_status" in issue for issue in issues)
