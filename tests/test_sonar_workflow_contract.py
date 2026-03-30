from __future__ import annotations

from pathlib import Path


def _read_workflow() -> str:
    return Path(".github/workflows/sonar.yml").read_text(encoding="utf-8")


def test_sonar_workflow_runs_on_push_without_repo_variable_gate() -> None:
    workflow = _read_workflow()

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "branches: [ main ]" in workflow
    assert "ENABLE_SONARQUBE_CLOUD_SCAN" not in workflow
    assert "github.actor != 'dependabot[bot]'" in workflow


def test_sonar_workflow_waits_for_quality_gate() -> None:
    workflow = _read_workflow()

    assert "-Dsonar.qualitygate.wait=true" in workflow


def test_contributing_and_readme_document_local_sonar_workflow() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "bash tools/run_sonar_local.sh --skip-scan" in contributing
    assert "python3 tools/fetch_sonar_issues.py --project-key skygazer42_pyimgano" in contributing
    assert "bash tools/run_sonar_local.sh --skip-scan" in readme
