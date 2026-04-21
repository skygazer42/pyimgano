from __future__ import annotations

from pathlib import Path


def _read_ci_workflow() -> str:
    return Path(".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_workflow_includes_deploy_smoke_job() -> None:
    workflow = _read_ci_workflow()

    assert "deploy_smoke:" in workflow
    assert "name: Deploy Smoke Path" in workflow
    assert "pyimgano-doctor --profile deploy-smoke --json" in workflow
    assert "pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset" in workflow
    assert "deploy_smoke_custom_cpu.json" in workflow
    assert (
        "pyimgano-train --config /tmp/deploy_smoke_ci.json --export-infer-config --export-deploy-bundle"
        in workflow
    )
    assert (
        "pyimgano validate-infer-config ./_deploy_smoke_run/deploy_bundle/infer_config.json"
        in workflow
    )
    assert "pyimgano bundle validate ./_deploy_smoke_run/deploy_bundle --json" in workflow
    assert "pyimgano runs acceptance ./_deploy_smoke_run --json" in workflow


def test_ci_workflow_quality_job_runs_deploy_smoke_docs_audit() -> None:
    workflow = _read_ci_workflow()

    assert "python tools/audit_deploy_smoke_docs.py" in workflow
    assert "python tools/audit_recipe_starters.py" in workflow
    assert "python tools/audit_release_checklist.py" in workflow


def test_ci_workflow_build_job_waits_for_deploy_smoke() -> None:
    workflow = _read_ci_workflow()

    assert "needs: [quality, test, deploy_smoke]" in workflow
