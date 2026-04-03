from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_doctor_extras_and_readiness() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "pyimgano-doctor" in text
    assert "--require-extras" in text
    assert "--recommend-extras" in text
    assert "--for-command" in text
    assert "--for-model" in text
    assert "starter_configs" in text
    assert "optional_baseline_count" in text
    assert "starter_list_command" in text
    assert "starter_info_command" in text
    assert "suggested_commands" in text
    assert "recipe_list_command" in text
    assert "recipe_info_command" in text
    assert "recipe_run_command" in text
    assert "dry_run_command" in text
    assert "preset_infer_command" in text
    assert "from_run_infer_command" in text
    assert "quality_command" in text
    assert "acceptance_command" in text
    assert "bundle_audit_command" in text
    assert "artifact_hints" in text
    assert "workflow_stage" in text
    assert "next_step_commands" in text
    assert "recommended_extra_profiles" in text
    assert "install_command" in text
    assert "model_info_command" in text
    assert "supports_pixel_map" in text
    assert "tested_runtime" in text
    assert "leaderboard_metadata.json" in text
    assert "embed.onnx" in text
    assert "embed.ts" in text
    assert "pyimgano train --recipe-info industrial-adapt --json" in text
    assert "pyimgano train --list-recipes" in text
    assert "--accelerators" in text
    assert "--run-dir" in text
    assert "--deploy-bundle" in text


def test_readme_mentions_doctor_require_extras_and_readiness() -> None:
    text = _read_text("README.md")

    assert "pyimgano-doctor" in text
    assert "--suite industrial-v4" in text
    assert "--recommend-extras" in text
    assert 'pip install "pyimgano[deploy]"' in text
    assert 'pip install "pyimgano[benchmark]"' in text
    assert 'pip install "pyimgano[tracking]"' in text
    assert 'pip install "pyimgano[cpu-offline]"' in text
    assert "--for-command train" in text
    assert "--for-command infer" in text
    assert "--for-command runs" in text
    assert "pyimgano-doctor --recommend-extras --for-command benchmark --json" in text
