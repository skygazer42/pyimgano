from __future__ import annotations

from pathlib import Path

from pyimgano.services.doctor_service import collect_doctor_payload


def test_collect_doctor_payload_returns_json_ready_shape() -> None:
    payload = collect_doctor_payload()

    assert payload["tool"] == "pyimgano-doctor"
    assert "optional_modules" in payload
    assert "baselines" in payload


def test_collect_doctor_payload_recommends_extras_for_export_onnx_command(
    monkeypatch,
) -> None:
    import pyimgano.services.doctor_service as doctor_service
    from pyimgano.workflow_guidance import artifact_hints_for_command
    from pyimgano.workflow_guidance import command_workflow_guidance
    from pyimgano.workflow_guidance import next_step_commands_for_command
    from pyimgano.workflow_guidance import suggested_commands_for_command
    from pyimgano.workflow_guidance import workflow_stage_for_command

    monkeypatch.setattr(
        doctor_service,
        "extra_installed",
        lambda extra: str(extra) not in {"onnx", "torch"},
    )

    payload = collect_doctor_payload(recommend_extras=True, for_command="export-onnx")
    guidance = command_workflow_guidance("export-onnx")

    recommendation = payload.get("extras_recommendation")
    assert isinstance(recommendation, dict)
    assert recommendation["target_kind"] == "command"
    assert recommendation["target"] == "export-onnx"
    assert recommendation["required_extras"] == ["onnx", "torch"]
    assert recommendation["workflow_stage"] == workflow_stage_for_command("export-onnx")
    assert recommendation["workflow_stage"] == guidance.workflow_stage
    assert recommendation["suggested_commands"] == suggested_commands_for_command("export-onnx")
    assert recommendation["suggested_commands"] == list(guidance.suggested_commands)
    assert recommendation["next_step_commands"] == next_step_commands_for_command("export-onnx")
    assert recommendation["next_step_commands"] == list(guidance.next_step_commands)
    assert recommendation["artifact_hints"] == artifact_hints_for_command("export-onnx")
    assert recommendation["artifact_hints"] == list(guidance.artifact_hints)
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
    from pyimgano.workflow_guidance import starter_benchmark_guidance
    from pyimgano.workflow_guidance import workflow_stage_for_command

    payload = collect_doctor_payload(recommend_extras=True, for_command="benchmark")
    guidance = starter_benchmark_guidance(default_starter_benchmark_name())

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
    assert recommendation["starter_list_command"] == guidance.list_command
    assert recommendation["starter_info_command"] == guidance.info_command
    assert recommendation["starter_run_command"] == guidance.run_command
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


def test_collect_doctor_payload_first_run_profile_exposes_guided_path() -> None:
    payload = collect_doctor_payload(profile="first-run")

    workflow_profile = payload.get("workflow_profile")
    assert isinstance(workflow_profile, dict)
    assert workflow_profile["profile"] == "first-run"
    assert workflow_profile["status"] == "ok"
    assert workflow_profile["offline_safe"] is True
    assert workflow_profile["target_kind"] == "profile"
    assert workflow_profile["required_modules"] == ["cv2", "numpy", "sklearn"]
    assert workflow_profile["missing_modules"] == []
    assert workflow_profile["required_extras"] == []
    assert workflow_profile["missing_extras"] == []
    assert workflow_profile["starter_commands"] == [
        "pyimgano-doctor --profile first-run --json",
        "pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained",
        "pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json",
        "pyimgano-benchmark --dataset custom --root ./_demo_custom_dataset --suite industrial-ci --resize 32 32 --limit-train 2 --limit-test 2 --no-pretrained --save-run --output-dir ./_demo_benchmark_run --suite-export csv",
        "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir ./_demo_custom_dataset/train/normal --input ./_demo_custom_dataset/test --save-jsonl ./_demo_results.jsonl",
        "pyimgano runs quality ./_demo_benchmark_run --json",
    ]
    assert workflow_profile["artifact_hints"] == [
        "./_demo_suite_run/report.json",
        "./_demo_benchmark_run/report.json",
        "./_demo_benchmark_run/leaderboard.csv",
        "./_demo_results.jsonl",
    ]
    readiness = payload.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness["target_kind"] == "profile"
    assert readiness["path"] == "first-run"
    assert readiness["status"] == "ok"


def test_collect_doctor_payload_benchmark_profile_uses_dataset_target_and_artifact_expectations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "custom"
    (root / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "test" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "test" / "anomaly").mkdir(parents=True, exist_ok=True)
    (root / "ground_truth" / "anomaly").mkdir(parents=True, exist_ok=True)
    (root / "train" / "normal" / "train_0.png").write_bytes(b"png")
    (root / "test" / "normal" / "good_0.png").write_bytes(b"png")
    (root / "test" / "anomaly" / "bad_0.png").write_bytes(b"png")
    (root / "ground_truth" / "anomaly" / "bad_0_mask.png").write_bytes(b"png")

    payload = collect_doctor_payload(profile="benchmark", dataset_target=str(root))

    workflow_profile = payload.get("workflow_profile")
    assert isinstance(workflow_profile, dict)
    assert workflow_profile["profile"] == "benchmark"
    assert workflow_profile["target_kind"] == "profile"
    assert workflow_profile["status"] == "warning"
    assert workflow_profile["starter_config"] == "official_mvtec_industrial_v4_cpu_offline.json"
    assert workflow_profile["required_extras"] == []
    assert workflow_profile["artifact_hints"] == [
        "leaderboard.csv",
        "best_by_baseline.csv",
        "skipped.csv",
        "leaderboard_metadata.json",
    ]
    assert workflow_profile["starter_commands"] == [
        "pyimgano benchmark --list-starter-configs",
        "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
        "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json",
    ]
    assert workflow_profile["next_step_commands"] == [
        "pyimgano-doctor --recommend-extras --for-command train --json",
        "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir /path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
        "pyimgano runs quality runs/<run_dir> --require-status audited --json",
    ]
    assert workflow_profile["dataset_target"] == str(root)
    assert "fewshot_train_set" in set(workflow_profile["issues"])

    readiness = payload.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness["target_kind"] == "profile"
    assert readiness["path"] == "benchmark"
    assert readiness["status"] == "warning"
    assert "fewshot_train_set" in set(readiness["issues"])


def test_collect_doctor_payload_deploy_profile_uses_run_readiness(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text('{"run_dir": "demo"}', encoding="utf-8")
    (run_dir / "config.json").write_text('{"recipe": "industrial-adapt"}', encoding="utf-8")
    (run_dir / "environment.json").write_text('{"python": "3.10"}', encoding="utf-8")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "infer_config.json").write_text(
        '{"model": {"name": "vision_patchcore", "model_kwargs": {}}, "defects": {"mask_format": "png"}}',
        encoding="utf-8",
    )

    payload = collect_doctor_payload(profile="deploy", run_dir=str(run_dir))

    workflow_profile = payload.get("workflow_profile")
    assert isinstance(workflow_profile, dict)
    assert workflow_profile["profile"] == "deploy"
    assert workflow_profile["status"] == "warning"
    assert workflow_profile["target_path"] == str(run_dir)
    assert workflow_profile["target_source"] == "run_dir"
    assert workflow_profile["artifact_hints"] == [
        "report.json",
        "config.json",
        "environment.json",
        "deploy_bundle/infer_config.json",
        "deploy_bundle/bundle_manifest.json",
    ]
    assert workflow_profile["starter_commands"] == [
        "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json",
        "pyimgano-doctor --profile deploy --deploy-bundle runs/<run_dir>/deploy_bundle --json",
        "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
    ]
    assert "insufficient_quality_status" in set(workflow_profile["issues"])

    readiness = payload.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness["target_kind"] == "profile"
    assert readiness["path"] == "deploy"
    assert readiness["status"] == "warning"


def test_collect_doctor_payload_publish_profile_exposes_publication_gate(tmp_path: Path) -> None:
    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    report = export_dir / "report.json"
    config = export_dir / "config.json"
    environment = export_dir / "environment.json"
    leaderboard = export_dir / "leaderboard.csv"
    metadata = export_dir / "leaderboard_metadata.json"

    report.write_text('{"suite": "industrial-v4"}', encoding="utf-8")
    config.write_text('{"config": {"seed": 123}}', encoding="utf-8")
    environment.write_text('{"fingerprint_sha256": "%s"}' % ("f" * 64), encoding="utf-8")
    leaderboard.write_text("name,auroc\nx,0.9\n", encoding="utf-8")

    import hashlib
    import json

    metadata.write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": hashlib.sha256(report.read_bytes()).hexdigest(),
                    "config_json": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "environment_json": hashlib.sha256(environment.read_bytes()).hexdigest(),
                },
                "exported_files": {
                    "leaderboard_csv": str(leaderboard),
                    "leaderboard_metadata_json": str(metadata),
                },
                "exported_file_digests": {
                    "leaderboard_csv": hashlib.sha256(leaderboard.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    payload = collect_doctor_payload(profile="publish", publication_target=str(export_dir))

    workflow_profile = payload.get("workflow_profile")
    assert isinstance(workflow_profile, dict)
    assert workflow_profile["profile"] == "publish"
    assert workflow_profile["status"] == "ok"
    assert workflow_profile["target_path"] == str(export_dir)
    assert workflow_profile["target_source"] == "publication_target"
    assert workflow_profile["artifact_hints"] == [
        "leaderboard.csv",
        "leaderboard_metadata.json",
        "report.json",
        "config.json",
        "environment.json",
    ]
    assert workflow_profile["starter_commands"] == [
        "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json",
        "pyimgano runs acceptance /path/to/suite_export --json",
        "pyimgano runs publication /path/to/suite_export --json",
    ]
    publication = payload.get("publication")
    assert isinstance(publication, dict)
    assert publication["status"] == "ready"
    assert publication["publication_ready"] is True

    readiness = payload.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness["target_kind"] == "profile"
    assert readiness["path"] == "publish"
    assert readiness["status"] == "ok"
