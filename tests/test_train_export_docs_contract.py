from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_train_export_infer_config_and_deploy_bundle() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--export-infer-config" in text
    assert "--export-deploy-bundle" in text
    assert "--list-starter-configs" in text
    assert "--starter-config-info" in text
    assert "optional_extras" in text
    assert "starter_tier" in text
    assert "starter_info_command" in text
    assert "starter_run_command" in text
    assert "deploy_bundle/" in text


def test_readme_mentions_export_deploy_bundle_validation_flow() -> None:
    text = _read_text("README.md")

    assert "--export-infer-config" in text
    assert "--export-deploy-bundle" in text
    assert "--list-starter-configs" in text
    assert "pyimgano-bundle validate" in text
