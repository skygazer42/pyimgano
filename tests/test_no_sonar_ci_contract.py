from __future__ import annotations

from pathlib import Path


def test_repository_no_longer_ships_sonar_ci_surface() -> None:
    removed_paths = [
        Path(".github/workflows/sonar.yml"),
        Path("Dockerfile.sonar"),
        Path("sonar-project.properties"),
        Path("tools/run_sonar_local.sh"),
        Path("tools/fetch_sonar_issues.py"),
    ]

    assert all(not path.exists() for path in removed_paths)


def test_contributor_docs_no_longer_reference_sonar_workflow() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    forbidden_needles = (
        "SonarCloud Workflow Checks",
        "tools/run_sonar_local.sh",
        "tools/fetch_sonar_issues.py",
        "Dockerfile.sonar",
        "For SonarCloud-related changes",
    )

    for needle in forbidden_needles:
        assert needle not in contributing
        assert needle not in readme
