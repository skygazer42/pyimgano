from pathlib import Path

from pyimgano.config import load_config
from pyimgano.workbench.config import WorkbenchConfig


def test_examples_configs_load_and_parse():
    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "examples" / "configs"

    paths = [
        configs_dir / "classical_colorhist_mahalanobis_cpu.json",
        configs_dir / "classical_edge_ecod_cpu.json",
        configs_dir / "classical_fft_lowfreq_ecod_cpu.json",
        configs_dir / "classical_hog_ecod_cpu.json",
        configs_dir / "classical_lbp_loop_cpu.json",
        configs_dir / "classical_patch_stats_ecod_cpu.json",
        configs_dir / "classical_structural_ecod_cpu.json",
        configs_dir / "deploy_smoke_custom_cpu.json",
        configs_dir / "industrial_adapt_fast.json",
        configs_dir / "industrial_adapt_audited.json",
        configs_dir / "industrial_embedding_core_fast.json",
        configs_dir / "industrial_adapt_highres.json",
        configs_dir / "industrial_adapt_defects_roi.json",
        configs_dir / "industrial_adapt_defects_fp40.json",
        configs_dir / "industrial_adapt_maps_tiling.json",
        configs_dir / "manifest_industrial_adapt_fast.json",
        configs_dir / "manifest_industrial_workflow_balanced.json",
        configs_dir / "micro_finetune_autoencoder.json",
    ]

    for p in paths:
        raw = load_config(p)
        cfg = WorkbenchConfig.from_dict(raw)
        assert isinstance(cfg.recipe, str)
        assert cfg.dataset.name
        assert cfg.model.name

        if p.name == "classical_colorhist_mahalanobis_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "classical_edge_ecod_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "classical_fft_lowfreq_ecod_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "classical_hog_ecod_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "classical_lbp_loop_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "classical_patch_stats_ecod_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "classical_structural_ecod_cpu.json":
            assert cfg.meta.purpose == "cpu-screening"
            assert cfg.meta.runtime_profile == "cpu-screening"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "deploy_smoke_custom_cpu.json":
            assert cfg.meta.purpose == "deploy-smoke"
            assert cfg.meta.runtime_profile == "cpu-offline"
            assert "deploy" in cfg.meta.required_extras
            assert "deploy_bundle/handoff_report.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_adapt_audited.json":
            assert cfg.meta.purpose == "audited-deploy"
            assert cfg.meta.runtime_profile == "gpu-audited"
            assert "deploy_bundle/handoff_report.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_embedding_core_fast.json":
            assert cfg.meta.purpose == "embedding-core"
            assert cfg.meta.runtime_profile == "gpu-embeddings"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_adapt_highres.json":
            assert cfg.meta.purpose == "highres-tiling"
            assert cfg.meta.runtime_profile == "gpu-highres"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_adapt_fast.json":
            assert cfg.meta.purpose == "fast-screening"
            assert cfg.meta.runtime_profile == "gpu-fast"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_adapt_defects_fp40.json":
            assert cfg.meta.purpose == "defects-fp40"
            assert cfg.meta.runtime_profile == "gpu-defects"
            assert "deploy_bundle/handoff_report.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_adapt_defects_roi.json":
            assert cfg.meta.purpose == "defects-roi"
            assert cfg.meta.runtime_profile == "gpu-defects"
            assert "deploy_bundle/handoff_report.json" in cfg.meta.expected_artifacts
        if p.name == "manifest_industrial_adapt_fast.json":
            assert cfg.meta.purpose == "manifest-fast"
            assert cfg.meta.runtime_profile == "manifest-fast"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "manifest_industrial_workflow_balanced.json":
            assert cfg.meta.purpose == "manifest-deploy"
            assert cfg.meta.runtime_profile == "manifest-balanced"
            assert "deploy_bundle/handoff_report.json" in cfg.meta.expected_artifacts
        if p.name == "industrial_adapt_maps_tiling.json":
            assert cfg.meta.purpose == "maps-tiling"
            assert cfg.meta.runtime_profile == "gpu-localization"
            assert "artifacts/infer_config.json" in cfg.meta.expected_artifacts
        if p.name == "micro_finetune_autoencoder.json":
            assert cfg.meta.purpose == "micro-finetune"
            assert cfg.meta.runtime_profile == "gpu-training"
            assert "checkpoints/model.pt" in cfg.meta.expected_artifacts
