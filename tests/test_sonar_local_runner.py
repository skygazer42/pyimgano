from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="tools/run_sonar_local.sh is exercised in the Linux Sonar workflow only.",
)


def _clean_runner_env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "SONAR_TOKEN",
        "PYIMGANO_SONAR_INSTALL_COMMAND",
        "PYIMGANO_SONAR_PYTEST_COMMAND",
        "PYIMGANO_SONAR_SCAN_COMMAND",
        "PYIMGANO_SONAR_PYTHON_BIN",
        "SONAR_SCANNER_IMAGE",
        "SONAR_HOST_URL",
        "SONAR_PROJECT_KEY",
    ):
        env.pop(key, None)
    env.update(overrides)
    return env


def test_run_sonar_local_dry_run_prints_pytest_and_scanner_steps() -> None:
    proc = subprocess.run(
        ["bash", "tools/run_sonar_local.sh", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=_clean_runner_env(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pip install -e .[dev,torch,skimage]" in proc.stdout
    assert "coverage erase" in proc.stdout
    assert "pytest -v --cov=pyimgano --cov-report= --cov-append tests/contracts" in proc.stdout
    assert (
        "pytest -v --cov=pyimgano --cov-report= --cov-append tests/test_[a-cA-C]*.py" in proc.stdout
    )
    assert "coverage xml" in proc.stdout
    assert "docker run --rm" in proc.stdout
    assert "-Dsonar.scanner.skipJreProvisioning=true" in proc.stdout


def test_run_sonar_local_dry_run_includes_project_version_when_provided() -> None:
    proc = subprocess.run(
        ["bash", "tools/run_sonar_local.sh", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=_clean_runner_env(SONAR_PROJECT_VERSION="1.2.3"),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "-Dsonar.projectVersion=1.2.3" in proc.stdout


def test_dockerfile_sonar_uses_local_runner_entrypoint() -> None:
    dockerfile = Path("Dockerfile.sonar").read_text(encoding="utf-8")

    assert "tools/run_sonar_local.sh" in dockerfile
    assert "ENTRYPOINT" in dockerfile


def test_dockerignore_excludes_local_virtualenvs_and_reports() -> None:
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert ".venv" in dockerignore
    assert ".git" in dockerignore
    assert "htmlcov" in dockerignore
    assert "coverage.xml" in dockerignore
    assert "_demo_custom_dataset" in dockerignore


def test_run_sonar_local_requires_token_when_scan_enabled() -> None:
    proc = subprocess.run(
        ["bash", "tools/run_sonar_local.sh"],
        capture_output=True,
        text=True,
        check=False,
        env=_clean_runner_env(),
    )

    assert proc.returncode == 1
    assert "SONAR_TOKEN is required" in proc.stderr


def test_run_sonar_local_skip_scan_runs_install_and_pytest_without_token() -> None:
    env = _clean_runner_env(
        PYIMGANO_SONAR_INSTALL_COMMAND="printf 'install-ok\\n'",
        PYIMGANO_SONAR_PYTEST_COMMAND="printf 'pytest-ok\\n'",
        PYIMGANO_SONAR_SCAN_COMMAND="printf 'scan-ok\\n'",
    )

    proc = subprocess.run(
        ["bash", "tools/run_sonar_local.sh", "--skip-scan"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "install-ok" in proc.stdout
    assert "pytest-ok" in proc.stdout
    assert "scan-ok" not in proc.stdout
