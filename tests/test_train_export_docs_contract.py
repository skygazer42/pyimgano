from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_train_export_infer_config_and_deploy_bundle() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--export-infer-config" in text
    assert "--export-deploy-bundle" in text
    assert "handoff_report.json" in text
    assert "--list-starter-configs" in text
    assert "--starter-config-info" in text
    assert "optional_extras" in text
    assert "starter_tier" in text
    assert "starter_info_command" in text
    assert "starter_run_command" in text
    assert "deploy_bundle/" in text
    assert "starter_status" in text
    assert "starter_reason" in text
    assert "manual-only" in text
    assert "generated-at-runtime" in text
    assert "Run command:" in text
    assert "Install hint:" in text
    assert "pyimgano train --list-recipes" in text
    assert "pyimgano train --recipe-info industrial-adapt --json" in text


def test_recipe_docs_cover_recipe_info_starter_metadata() -> None:
    text = _read_text("docs/RECIPES.md")

    assert "pyimgano-train --recipe-info industrial-adapt" in text
    assert "pyimgano-train --recipe-info anomalib-train" in text
    assert "pyimgano train --list-recipes" in text
    assert "pyimgano train --recipe-info industrial-adapt --json" in text
    assert "starter_status" in text
    assert "starter_reason" in text
    assert "manual-only" in text


def test_readme_mentions_export_deploy_bundle_validation_flow() -> None:
    text = _read_text("README.md")

    assert "--export-infer-config" in text
    assert "--export-deploy-bundle" in text
    assert "handoff_report.json" in text
    assert "--list-starter-configs" in text
    assert "pyimgano-bundle validate" in text


def test_readme_and_workbench_document_umbrella_train_discovery_commands() -> None:
    readme = _read_text("README.md")
    workbench = _read_text("docs/WORKBENCH.md")
    fastpath = _read_text("docs/INDUSTRIAL_FASTPATH.md")

    assert "pyimgano train --list-recipes" in readme
    assert "pyimgano train --recipe-info industrial-adapt --json" in readme
    assert "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json" in readme
    assert "pyimgano train --list-recipes" in workbench
    assert "pyimgano train --recipe-info industrial-adapt --json" in workbench
    assert "pyimgano train --list-recipes" in fastpath
    assert "pyimgano train --recipe-info industrial-adapt --json" in fastpath
    assert "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json" in fastpath
