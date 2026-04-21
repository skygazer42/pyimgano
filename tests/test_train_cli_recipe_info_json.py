import json


def test_train_cli_recipe_info_json_includes_callable_path(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "industrial-adapt"
    assert parsed["callable"].endswith("pyimgano.recipes.builtin.industrial_adapt.industrial_adapt")
    assert parsed["metadata"]["default_config"] == "examples/configs/deploy_smoke_custom_cpu.json"
    assert parsed["metadata"]["starter_configs"] == [
        "examples/configs/deploy_smoke_custom_cpu.json",
        "examples/configs/industrial_adapt_audited.json",
        "examples/configs/manifest_industrial_workflow_balanced.json",
    ]
    assert (
        parsed["run_command"]
        == "pyimgano train --config examples/configs/deploy_smoke_custom_cpu.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "cpu-offline"
    assert "deploy_bundle/handoff_report.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_fp40_starter_configs(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt-fp40", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "industrial-adapt-fp40"
    assert (
        parsed["metadata"]["default_config"]
        == "examples/configs/industrial_adapt_defects_fp40.json"
    )
    assert parsed["metadata"]["starter_configs"] == [
        "examples/configs/industrial_adapt_defects_fp40.json",
        "examples/configs/industrial_adapt_defects_roi.json",
    ]
    assert parsed["metadata"]["runtime_profile"] == "gpu-defects"


def test_train_cli_recipe_info_json_surfaces_highres_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt-highres", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "industrial-adapt-highres"
    assert parsed["metadata"]["default_config"] == "examples/configs/industrial_adapt_highres.json"
    assert parsed["metadata"]["runtime_profile"] == "gpu-highres"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_structural_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-structural-ecod", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-structural-ecod"
    assert (
        parsed["metadata"]["default_config"]
        == "examples/configs/classical_structural_ecod_cpu.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_edge_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-edge-ecod", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-edge-ecod"
    assert parsed["metadata"]["default_config"] == "examples/configs/classical_edge_ecod_cpu.json"
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_colorhist_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-colorhist-mahalanobis", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-colorhist-mahalanobis"
    assert (
        parsed["metadata"]["default_config"]
        == "examples/configs/classical_colorhist_mahalanobis_cpu.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_fft_lowfreq_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-fft-lowfreq-ecod", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-fft-lowfreq-ecod"
    assert (
        parsed["metadata"]["default_config"]
        == "examples/configs/classical_fft_lowfreq_ecod_cpu.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_hog_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-hog-ecod", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-hog-ecod"
    assert parsed["metadata"]["default_config"] == "examples/configs/classical_hog_ecod_cpu.json"
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_lbp_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-lbp-loop", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-lbp-loop"
    assert parsed["metadata"]["default_config"] == "examples/configs/classical_lbp_loop_cpu.json"
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_classical_patch_stats_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-patch-stats-ecod", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-patch-stats-ecod"
    assert (
        parsed["metadata"]["default_config"]
        == "examples/configs/classical_patch_stats_ecod_cpu.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "cpu-screening"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_micro_finetune_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "micro-finetune-autoencoder", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "micro-finetune-autoencoder"
    assert (
        parsed["metadata"]["default_config"] == "examples/configs/micro_finetune_autoencoder.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "gpu-training"
    assert "checkpoints/model.pt" in parsed["metadata"]["expected_artifacts"]


def test_train_cli_recipe_info_json_surfaces_placeholder_recipe_starter_status(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "anomalib-train", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "anomalib-train"
    assert parsed["install_hint"] == "pip install 'pyimgano[anomalib]'"
    assert parsed["metadata"]["starter_status"] == "manual-only"
    assert "anomalib" in parsed["metadata"]["starter_reason"].lower()


def test_train_cli_recipe_info_json_surfaces_generated_recipe_starter_status(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "classical-struct-iforest-synth", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "classical-struct-iforest-synth"
    assert parsed["metadata"]["starter_status"] == "generated-at-runtime"
    assert "synthetic dataset" in parsed["metadata"]["starter_reason"].lower()


def test_train_cli_recipe_info_json_surfaces_embedding_core_default_config(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-embedding-core-fast", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["name"] == "industrial-embedding-core-fast"
    assert (
        parsed["metadata"]["default_config"]
        == "examples/configs/industrial_embedding_core_fast.json"
    )
    assert parsed["metadata"]["runtime_profile"] == "gpu-embeddings"
    assert "artifacts/infer_config.json" in parsed["metadata"]["expected_artifacts"]
