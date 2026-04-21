from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cli_reference_documents_pyim_starter_pick_metadata() -> None:
    text = _read_text("docs/CLI_REFERENCE.md")

    assert "--goal" in text
    assert "--objective" in text
    assert "--selection-profile" in text
    assert "--topk" in text
    assert "supports_pixel_map" in text
    assert "tested_runtime" in text
    assert "why_this_pick" in text
    assert "install hint" in text
    assert "Selection Context" in text
    assert "Goal Context" in text
    assert "Goal Picks" in text
    assert "Suggested Commands" in text
    assert "recipe_list_command" in text
    assert "recipe_info_command" in text
    assert "dry_run_command" in text
    assert "preflight_command" in text
    assert "recipe_run_command" in text


def test_algorithm_selection_guide_and_examples_document_pyim_goals() -> None:
    guide = _read_text("docs/ALGORITHM_SELECTION_GUIDE.md")
    cli_reference = _read_text("docs/CLI_REFERENCE.md")
    examples = _read_text("examples/README.md")

    assert "pyim --goal first-run --json" in guide
    assert "pyim --goal deployable --json" in guide
    assert "deploy_smoke_custom_cpu.json" in guide
    assert "classical_colorhist_mahalanobis_cpu.json" in guide
    assert "classical_edge_ecod_cpu.json" in guide
    assert "classical_fft_lowfreq_ecod_cpu.json" in guide
    assert "classical_hog_ecod_cpu.json" in guide
    assert "classical_lbp_loop_cpu.json" in guide
    assert "classical_patch_stats_ecod_cpu.json" in guide
    assert "classical_structural_ecod_cpu.json" in guide
    assert "industrial_adapt_audited.json" in guide
    assert "manifest_industrial_workflow_balanced.json" in guide
    assert "industrial_adapt_defects_fp40.json" in guide
    assert "industrial_adapt_defects_roi.json" in guide
    assert "industrial_adapt_maps_tiling.json" in guide
    assert "classical_colorhist_mahalanobis_cpu.json" in cli_reference
    assert "classical_edge_ecod_cpu.json" in cli_reference
    assert "classical_fft_lowfreq_ecod_cpu.json" in cli_reference
    assert "classical_hog_ecod_cpu.json" in cli_reference
    assert "classical_lbp_loop_cpu.json" in cli_reference
    assert "classical_patch_stats_ecod_cpu.json" in cli_reference
    assert "classical_structural_ecod_cpu.json" in cli_reference
    assert "industrial_adapt_audited.json" in cli_reference
    assert "manifest_industrial_workflow_balanced.json" in cli_reference
    assert "industrial_adapt_defects_fp40.json" in cli_reference
    assert "industrial_adapt_defects_roi.json" in cli_reference
    assert "industrial_adapt_maps_tiling.json" in cli_reference
    assert "pyim --goal first-run --json" in examples
    assert "classical_colorhist_mahalanobis_cpu.json" in examples
    assert "classical_edge_ecod_cpu.json" in examples
    assert "classical_fft_lowfreq_ecod_cpu.json" in examples
    assert "classical_hog_ecod_cpu.json" in examples
    assert "classical_lbp_loop_cpu.json" in examples
    assert "classical_patch_stats_ecod_cpu.json" in examples
    assert "classical_structural_ecod_cpu.json" in examples
    assert "industrial_adapt_defects_fp40.json" in examples
    assert "industrial_adapt_defects_roi.json" in examples
    assert "industrial_adapt_maps_tiling.json" in examples
    assert "baseline" in examples.lower()
    assert "optional backend" in examples.lower()


def test_algorithm_selection_guide_documents_anomalydino_custom_embedder_checkpoint_limit() -> None:
    guide = _read_text("docs/ALGORITHM_SELECTION_GUIDE.md")

    assert "pass a custom embedder for offline usage" in guide
    assert "Custom embedders are runtime-only and are not serialized into checkpoints" in guide
    assert "skip checkpoint artifacts" in guide
    assert "TorchHubDinoV2Embedder" in guide
