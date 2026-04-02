from __future__ import annotations

from pathlib import Path


def _read_publish_workflow() -> str:
    return Path(".github/workflows/publish.yml").read_text(encoding="utf-8")


def test_publish_workflow_includes_release_readiness_gate() -> None:
    workflow = _read_publish_workflow()

    assert "release-readiness:" in workflow
    assert "name: Release Readiness" in workflow
    assert "python tools/audit_release_surface.py" in workflow
    assert "python tools/audit_adoption_docs.py" in workflow
    assert "python tools/audit_deploy_smoke_docs.py" in workflow
    assert "python tools/audit_recipe_starters.py" in workflow
    assert "python tools/audit_release_checklist.py" in workflow
    assert "pyimgano-doctor --profile deploy-smoke --json" in workflow
    assert "deploy_smoke_custom_cpu.json" in workflow
    assert "pyimgano bundle validate ./_release_deploy_smoke_run/deploy_bundle --json" in workflow
    assert (
        "pyimgano runs acceptance ./_release_deploy_smoke_run --require-status audited --check-bundle-hashes --json"
        in workflow
    )


def test_publish_workflow_build_job_waits_for_release_readiness() -> None:
    workflow = _read_publish_workflow()

    assert "needs: [release-readiness]" in workflow


def test_publishing_doc_mentions_release_readiness_job() -> None:
    publishing = Path("docs/PUBLISHING.md").read_text(encoding="utf-8")

    assert "Release Readiness" in publishing
    assert "publish.yml" in publishing
    assert "deploy-smoke" in publishing
