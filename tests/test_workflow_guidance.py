from __future__ import annotations


def test_workflow_guidance_exposes_expected_stage_order() -> None:
    from pyimgano.workflow_guidance import list_workflow_stages

    stages = list_workflow_stages()
    assert [stage.key for stage in stages] == [
        "discover",
        "benchmark",
        "train",
        "export",
        "infer",
        "validate",
        "gate",
    ]


def test_workflow_guidance_includes_export_recommendation_commands() -> None:
    from pyimgano.workflow_guidance import list_workflow_stages

    stages = {stage.key: stage for stage in list_workflow_stages()}
    train = stages["train"]
    export = stages["export"]

    assert list(train.commands) == [
        "pyimgano doctor --recommend-extras --for-command train --json",
        "pyimgano train --list-recipes",
        "pyimgano train --recipe-info industrial-adapt --json",
        "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
        "pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json",
        "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
    ]
    assert export.title == "Export"
    assert list(export.commands) == [
        "pyimgano doctor --recommend-extras --for-command export-onnx --json",
        "pyimgano doctor --recommend-extras --for-command export-torchscript --json",
    ]


def test_workflow_guidance_exposes_command_stage_and_next_steps() -> None:
    from pyimgano.workflow_guidance import artifact_hints_for_command
    from pyimgano.workflow_guidance import command_workflow_guidance
    from pyimgano.workflow_guidance import model_info_command_for_model
    from pyimgano.workflow_guidance import next_step_commands_for_model
    from pyimgano.workflow_guidance import suggested_commands_for_model
    from pyimgano.workflow_guidance import workflow_stage_for_model
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import suggested_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    assert workflow_stage_for_command("benchmark") == "benchmark"
    assert workflow_stage_for_command("train") == "train"
    assert workflow_stage_for_command("export-onnx") == "validate"
    assert workflow_stage_for_command("infer") == "infer"
    assert workflow_stage_for_command("runs") == "gate"

    assert next_step_commands_for_command("benchmark") == [
        "pyimgano-doctor --recommend-extras --for-command train --json",
    ]
    assert suggested_commands_for_command("benchmark") == [
        "pyimgano benchmark --list-starter-configs",
        "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
        "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json",
    ]
    assert artifact_hints_for_command("benchmark") == [
        "leaderboard.csv",
        "best_by_baseline.csv",
        "skipped.csv",
        "leaderboard_metadata.json",
    ]
    assert next_step_commands_for_command("train") == [
        "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
    ]
    assert next_step_commands_for_command("export-onnx") == [
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]
    assert next_step_commands_for_command("infer") == [
        "pyimgano-doctor --recommend-extras --for-command runs --json",
    ]
    assert next_step_commands_for_command("runs") == [
        "pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json",
    ]
    assert suggested_commands_for_command("train") == [
        "pyimgano train --list-recipes",
        "pyimgano train --recipe-info industrial-adapt --json",
        "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
        "pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json",
        "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
        "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
    ]
    assert suggested_commands_for_command("export-onnx") == [
        "pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained",
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]
    assert artifact_hints_for_command("infer") == [
        "results.jsonl",
        "masks/ (optional)",
        "overlays/ (optional)",
        "regions.jsonl (optional)",
    ]
    assert workflow_stage_for_model("vision_openclip_patch_map") == "discover"
    assert (
        model_info_command_for_model("vision_openclip_patch_map")
        == "pyimgano-benchmark --model-info vision_openclip_patch_map --json"
    )
    assert suggested_commands_for_model("vision_openclip_patch_map") == [
        "pyimgano-benchmark --model-info vision_openclip_patch_map --json",
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]
    assert next_step_commands_for_model("vision_openclip_patch_map") == [
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]
    infer_guidance = command_workflow_guidance("infer")
    assert infer_guidance.target_kind == "command"
    assert infer_guidance.target == "infer"
    assert infer_guidance.workflow_stage == "infer"
    assert list(infer_guidance.suggested_commands) == [
        "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir /path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
        "pyimgano-infer --from-run runs/<run_dir> --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
    ]


def test_workflow_guidance_exposes_structured_model_guidance() -> None:
    from pyimgano.workflow_guidance import model_workflow_guidance

    guidance = model_workflow_guidance("vision_openclip_patch_map")

    assert guidance.workflow_stage == "discover"
    assert (
        guidance.model_info_command
        == "pyimgano-benchmark --model-info vision_openclip_patch_map --json"
    )
    assert guidance.target_kind == "model"
    assert guidance.target == "vision_openclip_patch_map"
    assert list(guidance.suggested_commands) == [
        "pyimgano-benchmark --model-info vision_openclip_patch_map --json",
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]
    assert list(guidance.next_step_commands) == [
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]


def test_workflow_guidance_exposes_root_help_command_groups() -> None:
    from pyimgano.workflow_guidance import artifact_acceptance_commands
    from pyimgano.workflow_guidance import benchmark_publication_commands
    from pyimgano.workflow_guidance import industrial_fast_path_commands

    assert industrial_fast_path_commands() == [
        "pyimgano doctor --recommend-extras --for-command train --json",
        "pyimgano doctor --recommend-extras --for-command infer --json",
        "pyimgano doctor --recommend-extras --for-command runs --json",
        "pyimgano train --list-recipes",
        "pyimgano train --recipe-info industrial-adapt --json",
        "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
        "pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json",
        "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
        "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
        "pyimgano bundle run runs/<run_dir>/deploy_bundle --image-dir /path/to/images --output-dir ./bundle_run --json",
        "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
        "pyimgano runs quality runs/<run_dir> --require-status audited --json",
        "pyimgano runs acceptance runs/<run_dir> --require-status audited --json",
    ]
    assert benchmark_publication_commands() == [
        "pyimgano benchmark --list-starter-configs",
        "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
        "pyimgano benchmark --list-official-configs",
        "pyimgano benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
        "pyimgano runs acceptance /path/to/suite_export --json",
        "pyimgano runs publication /path/to/suite_export --json",
    ]
    assert artifact_acceptance_commands() == [
        "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
        "pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json",
    ]


def test_workflow_guidance_exposes_deploy_smoke_starter_path() -> None:
    from pyimgano.workflow_guidance import starter_path_by_name

    deploy_smoke = starter_path_by_name("deploy-smoke")

    assert deploy_smoke is not None
    assert deploy_smoke.title == "Deployment Smoke Path"
    assert list(deploy_smoke.commands) == [
        "pyimgano-doctor --profile deploy-smoke --json",
        "pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained",
        "pyimgano-train --config examples/configs/deploy_smoke_custom_cpu.json --root ./_demo_custom_dataset --export-infer-config --export-deploy-bundle",
        "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
        "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
        "pyimgano runs quality runs/<run_dir> --json",
    ]


def test_workflow_guidance_exposes_shared_starter_benchmark_commands() -> None:
    from pyimgano.workflow_guidance import default_starter_benchmark_name
    from pyimgano.workflow_guidance import starter_benchmark_guidance
    from pyimgano.workflow_guidance import starter_benchmark_info_command
    from pyimgano.workflow_guidance import starter_benchmark_list_command
    from pyimgano.workflow_guidance import starter_benchmark_run_command

    assert default_starter_benchmark_name() == "official_mvtec_industrial_v4_cpu_offline.json"
    assert starter_benchmark_list_command() == "pyimgano benchmark --list-starter-configs"
    assert (
        starter_benchmark_info_command()
        == "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json"
    )
    assert (
        starter_benchmark_run_command()
        == "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json"
    )
    guidance = starter_benchmark_guidance()
    assert guidance.target_kind == "starter-benchmark"
    assert guidance.target == "official_mvtec_industrial_v4_cpu_offline.json"
    assert guidance.list_command == starter_benchmark_list_command()
    assert guidance.info_command == starter_benchmark_info_command()
    assert guidance.run_command == starter_benchmark_run_command()


def test_workflow_guidance_exposes_first_ten_minutes_path() -> None:
    from pyimgano.workflow_guidance import first_ten_minutes_commands

    assert first_ten_minutes_commands() == [
        "pyimgano-doctor --profile first-run --json",
        "pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained",
        "pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json",
        "pyimgano-benchmark --dataset custom --root ./_demo_custom_dataset --suite industrial-ci --resize 32 32 --limit-train 2 --limit-test 2 --no-pretrained --save-run --output-dir ./_demo_benchmark_run --suite-export csv",
        "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir ./_demo_custom_dataset/train/normal --input ./_demo_custom_dataset/test --save-jsonl ./_demo_results.jsonl",
        "pyimgano runs quality ./_demo_benchmark_run --json",
    ]


def test_workflow_guidance_exposes_named_starter_paths() -> None:
    from pyimgano.workflow_guidance import list_starter_paths
    from pyimgano.workflow_guidance import starter_path_by_name

    paths = list_starter_paths()
    assert [path.name for path in paths] == [
        "deploy-smoke",
        "first-run",
        "benchmark",
        "deploy",
        "publish",
    ]

    deploy_smoke = starter_path_by_name("deploy-smoke")
    assert deploy_smoke is not None
    assert deploy_smoke.title == "Deployment Smoke Path"
    assert list(deploy_smoke.commands)[0] == "pyimgano-doctor --profile deploy-smoke --json"

    first_run = starter_path_by_name("first-run")
    assert first_run is not None
    assert first_run.title == "First 10 Minutes"
    assert list(first_run.commands)[0] == "pyimgano-doctor --profile first-run --json"

    publish = starter_path_by_name("publish")
    assert publish is not None
    assert publish.title == "Publication Gate"
    assert list(publish.commands) == [
        "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json",
        "pyimgano runs acceptance /path/to/suite_export --json",
        "pyimgano runs publication /path/to/suite_export --json",
    ]
