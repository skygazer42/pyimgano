from __future__ import annotations

from pyimgano.services.doctor_service import collect_doctor_payload


def test_collect_doctor_payload_returns_json_ready_shape() -> None:
    payload = collect_doctor_payload()

    assert payload["tool"] == "pyimgano-doctor"
    assert "optional_modules" in payload
    assert "baselines" in payload


def test_collect_doctor_payload_recommends_extras_for_export_onnx_command() -> None:
    from pyimgano.workflow_guidance import artifact_hints_for_command
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import suggested_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="export-onnx")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target_kind"] == "command"
    assert recommendation["target"] == "export-onnx"
    assert recommendation["required_extras"] == ["onnx", "torch"]
    assert recommendation["workflow_stage"] == workflow_stage_for_command("export-onnx")
    assert recommendation["suggested_commands"] == suggested_commands_for_command("export-onnx")
    assert recommendation["next_step_commands"] == next_step_commands_for_command("export-onnx")
    assert recommendation["artifact_hints"] == artifact_hints_for_command("export-onnx")
    assert recommendation["install_hint"] == "pip install 'pyimgano[onnx,torch]'"


def test_collect_doctor_payload_recommends_extras_for_export_torchscript_command() -> None:
    from pyimgano.workflow_guidance import artifact_hints_for_command
    from pyimgano.workflow_guidance import suggested_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="export-torchscript")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target"] == "export-torchscript"
    assert recommendation["required_extras"] == ["torch"]
    assert recommendation["workflow_stage"] == workflow_stage_for_command("export-torchscript")
    assert recommendation["suggested_commands"] == suggested_commands_for_command("export-torchscript")
    assert recommendation["artifact_hints"] == artifact_hints_for_command("export-torchscript")


def test_collect_doctor_payload_recommends_train_commands() -> None:
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="train")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target"] == "train"
    assert recommendation["required_extras"] == ["torch"]
    assert recommendation["workflow_stage"] == workflow_stage_for_command("train")
    assert recommendation["suggested_commands"] == [
        "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
        "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
    ]
    assert recommendation["next_step_commands"] == next_step_commands_for_command("train")
    assert recommendation["artifact_hints"] == [
        "artifacts/infer_config.json",
        "artifacts/calibration_card.json",
        "deploy_bundle/infer_config.json",
        "deploy_bundle/bundle_manifest.json",
    ]


def test_collect_doctor_payload_recommends_infer_commands() -> None:
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="infer")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target"] == "infer"
    assert recommendation["recommended_extras"] == ["onnx", "openvino", "torch"]
    assert recommendation["workflow_stage"] == workflow_stage_for_command("infer")
    assert recommendation["suggested_commands"] == [
        "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir /path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
        "pyimgano-infer --from-run runs/<run_dir> --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
    ]
    assert recommendation["next_step_commands"] == next_step_commands_for_command("infer")
    assert recommendation["artifact_hints"] == [
        "results.jsonl",
        "masks/ (optional)",
        "overlays/ (optional)",
        "regions.jsonl (optional)",
    ]


def test_collect_doctor_payload_recommends_runs_commands() -> None:
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="runs")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target"] == "runs"
    assert recommendation["required_extras"] == []
    assert recommendation["workflow_stage"] == workflow_stage_for_command("runs")
    assert recommendation["suggested_commands"] == [
        "pyimgano runs quality runs/<run_dir> --require-status audited --json",
        "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
    ]
    assert recommendation["next_step_commands"] == next_step_commands_for_command("runs")
    assert recommendation["artifact_hints"] == [
        "report.json",
        "config.json",
        "environment.json",
        "leaderboard_metadata.json (suite exports)",
    ]


def test_collect_doctor_payload_recommends_starter_suite_extras_for_benchmark_command() -> None:
    from pyimgano.workflow_guidance import artifact_hints_for_command
    from pyimgano.workflow_guidance import default_starter_benchmark_name
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import suggested_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="benchmark")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target_kind"] == "command"
    assert recommendation["target"] == "benchmark"
    assert recommendation["required_extras"] == []
    assert recommendation["recommended_extras"] == ["clip", "skimage", "torch"]
    assert recommendation["workflow_stage"] == workflow_stage_for_command("benchmark")
    assert recommendation["optional_baseline_count"] == 11
    assert recommendation["starter_configs"] == [
        "official_manifest_industrial_v4_cpu_offline.json",
        "official_mvtec_industrial_v4_cpu_offline.json",
        "official_visa_industrial_v4_cpu_offline.json",
    ]
    assert recommendation["starter_list_command"] == "pyimgano benchmark --list-starter-configs"
    assert (
        recommendation["starter_info_command"]
        == "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json"
    )
    assert (
        recommendation["starter_run_command"]
        == f"pyimgano-benchmark --config {default_starter_benchmark_name()}"
    )
    assert recommendation["suggested_commands"] == suggested_commands_for_command("benchmark")
    assert recommendation["next_step_commands"] == next_step_commands_for_command("benchmark")
    assert recommendation["artifact_hints"] == artifact_hints_for_command("benchmark")


def test_collect_doctor_payload_recommends_extras_for_model(monkeypatch) -> None:
    import pyimgano.services.doctor_service as doctor_service
    from pyimgano.workflow_guidance import model_workflow_guidance
    from pyimgano.workflow_guidance import model_info_command_for_model

    monkeypatch.setattr(
        doctor_service,
        "extra_installed",
        lambda extra: str(extra) not in {"torch", "clip"},
    )

    payload = collect_doctor_payload(recommend_extras=True, for_model="vision_openclip_patch_map")
    guidance = model_workflow_guidance("vision_openclip_patch_map")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target_kind"] == "model"
    assert recommendation["target"] == "vision_openclip_patch_map"
    assert recommendation["workflow_stage"] == guidance.workflow_stage
    assert recommendation["required_extras"] == ["clip", "torch"]
    assert recommendation["missing_extras"] == ["clip", "torch"]
    assert recommendation["supports_pixel_map"] is True
    assert recommendation["tested_runtime"] == "torch"
    assert recommendation["model_info_command"] == model_info_command_for_model("vision_openclip_patch_map")
    assert recommendation["suggested_commands"] == list(guidance.suggested_commands)
    assert recommendation["next_step_commands"] == list(guidance.next_step_commands)
    assert recommendation["install_hint"] == "pip install 'pyimgano[clip,torch]'"
