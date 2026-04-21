from __future__ import annotations


def test_format_suite_check_line_renders_missing_extras_suffix() -> None:
    from pyimgano.doctor_rendering import format_suite_check_line

    line = format_suite_check_line(
        suite_name="industrial-v4",
        info={
            "summary": {
                "total": 5,
                "runnable": 3,
                "missing_extras": ["torch", "faiss"],
            }
        },
    )

    assert line == "- industrial-v4: runnable 3/5 (missing extras: torch, faiss)"


def test_format_require_extras_line_handles_missing_and_ok() -> None:
    from pyimgano.doctor_rendering import format_require_extras_line

    assert (
        format_require_extras_line(
            {
                "required": ["torch"],
                "missing": [],
                "ok": True,
                "install_hint": None,
            }
        )
        == "require_extras: OK"
    )

    assert (
        format_require_extras_line(
            {
                "required": ["torch", "faiss"],
                "missing": ["faiss"],
                "ok": False,
                "install_hint": "pip install 'pyimgano[faiss]'",
            }
        )
        == "require_extras: MISSING (faiss) -> pip install 'pyimgano[faiss]'"
    )


def test_format_readiness_lines_renders_status_and_issues() -> None:
    from pyimgano.doctor_rendering import format_readiness_lines

    lines = format_readiness_lines(
        {
            "target_kind": "run",
            "path": "/tmp/run_a",
            "status": "warning",
            "issues": ["insufficient_quality_status", "missing_bundle_manifest"],
        }
    )

    assert lines == [
        "readiness:",
        "- target_kind: run",
        "- path: /tmp/run_a",
        "- status: warning",
        "- issues: insufficient_quality_status, missing_bundle_manifest",
    ]


def test_format_extra_recommendation_lines_renders_install_hint_and_missing() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "command",
            "target": "export-onnx",
            "workflow_stage": "validate",
            "required_extras": ["torch", "onnx"],
            "recommended_extras": [],
            "export_command": "pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained",
            "infer_followup_command": "pyimgano-doctor --recommend-extras --for-command infer --json",
            "suggested_commands": [
                "pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained",
                "pyimgano-doctor --recommend-extras --for-command infer --json",
            ],
            "next_step_commands": [
                "pyimgano-doctor --recommend-extras --for-command infer --json",
            ],
            "artifact_hints": [
                "embed.onnx",
                "onnx sweep JSON (optional)",
            ],
            "missing_extras": ["torch", "onnx"],
            "available_extras": [],
            "install_hint": "pip install 'pyimgano[onnx,torch]'",
            "notes": ["Exports require torch plus ONNX tooling."],
        }
    )

    assert lines == [
        "extras_recommendation:",
        "- target_kind: command",
        "- target: export-onnx",
        "- workflow_stage: validate",
        "- required_extras: torch, onnx",
        "- export_command: pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained",
        "- infer_followup_command: pyimgano-doctor --recommend-extras --for-command infer --json",
        "- suggested_commands: pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained; pyimgano-doctor --recommend-extras --for-command infer --json",
        "- next_step_commands: pyimgano-doctor --recommend-extras --for-command infer --json",
        "- artifact_hints: embed.onnx; onnx sweep JSON (optional)",
        "- missing_extras: torch, onnx",
        "- install_hint: pip install 'pyimgano[onnx,torch]'",
        "- notes: Exports require torch plus ONNX tooling.",
    ]


def test_format_extra_recommendation_lines_renders_starter_benchmark_context() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "command",
            "target": "benchmark",
            "workflow_stage": "benchmark",
            "required_extras": [],
            "recommended_extras": ["clip", "skimage", "torch"],
            "optional_baseline_count": 11,
            "starter_configs": [
                "official_manifest_industrial_v4_cpu_offline.json",
                "official_mvtec_industrial_v4_cpu_offline.json",
            ],
            "starter_list_command": "pyimgano benchmark --list-starter-configs",
            "starter_info_command": "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
            "starter_run_command": "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json",
            "suggested_commands": [
                "pyimgano benchmark --list-starter-configs",
                "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
            ],
            "next_step_commands": [
                "pyimgano-doctor --recommend-extras --for-command train --json",
            ],
            "artifact_hints": [
                "leaderboard.csv",
                "leaderboard_metadata.json",
            ],
            "install_hint": "pip install 'pyimgano[clip,skimage,torch]'",
        }
    )

    assert "- recommended_extras: clip, skimage, torch" in lines
    assert "- workflow_stage: benchmark" in lines
    assert "- optional_baseline_count: 11" in lines
    assert (
        "- starter_configs: official_manifest_industrial_v4_cpu_offline.json, official_mvtec_industrial_v4_cpu_offline.json"
        in lines
    )
    assert "- starter_list_command: pyimgano benchmark --list-starter-configs" in lines
    assert (
        "- starter_info_command: pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json"
        in lines
    )
    assert (
        "- starter_run_command: pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json"
        in lines
    )
    assert (
        "- suggested_commands: pyimgano benchmark --list-starter-configs; pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json"
        in lines
    )
    assert (
        "- next_step_commands: pyimgano-doctor --recommend-extras --for-command train --json"
        in lines
    )
    assert "- artifact_hints: leaderboard.csv; leaderboard_metadata.json" in lines


def test_format_extra_recommendation_lines_renders_suggested_commands() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "command",
            "target": "train",
            "required_extras": ["torch"],
            "recommended_extras": [],
            "recipe_list_command": "pyimgano train --list-recipes",
            "recipe_info_command": "pyimgano train --recipe-info industrial-adapt --json",
            "dry_run_command": "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
            "preflight_command": (
                "pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json"
            ),
            "recipe_run_command": (
                "pyimgano train --config examples/configs/industrial_adapt_audited.json "
                "--export-infer-config --export-deploy-bundle"
            ),
            "suggested_commands": [
                "pyimgano train --list-recipes",
                "pyimgano train --recipe-info industrial-adapt --json",
                "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
                "pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json",
                "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
                "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
            ],
        }
    )

    assert "- recipe_list_command: pyimgano train --list-recipes" in lines
    assert "- recipe_info_command: pyimgano train --recipe-info industrial-adapt --json" in lines
    assert (
        "- dry_run_command: pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json"
        in lines
    )
    assert (
        "- preflight_command: pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json"
        in lines
    )
    assert (
        "- recipe_run_command: pyimgano train --config examples/configs/industrial_adapt_audited.json "
        "--export-infer-config --export-deploy-bundle" in lines
    )
    assert (
        "- suggested_commands: pyimgano train --list-recipes; pyimgano train --recipe-info industrial-adapt --json; pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json; pyimgano train --preflight --config examples/configs/industrial_adapt_audited.json --json; pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle; pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json"
        in lines
    )


def test_format_extra_recommendation_lines_renders_artifact_hints() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "command",
            "target": "infer",
            "artifact_hints": [
                "results.jsonl",
                "masks/ (optional)",
            ],
        }
    )

    assert "- artifact_hints: results.jsonl; masks/ (optional)" in lines


def test_format_extra_recommendation_lines_renders_infer_structured_commands() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "command",
            "target": "infer",
            "preset_infer_command": (
                "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir "
                "/path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl"
            ),
            "from_run_infer_command": (
                "pyimgano-infer --from-run runs/<run_dir> --input /path/to/images "
                "--save-jsonl /tmp/pyimgano_results.jsonl"
            ),
        }
    )

    assert (
        "- preset_infer_command: pyimgano-infer --model-preset industrial-template-ncc-map "
        "--train-dir /path/to/train/normal --input /path/to/images "
        "--save-jsonl /tmp/pyimgano_results.jsonl" in lines
    )
    assert (
        "- from_run_infer_command: pyimgano-infer --from-run runs/<run_dir> --input /path/to/images "
        "--save-jsonl /tmp/pyimgano_results.jsonl" in lines
    )


def test_format_extra_recommendation_lines_renders_model_recommendation_context() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "model",
            "target": "vision_openclip_patch_map",
            "workflow_stage": "discover",
            "required_extras": ["clip", "torch"],
            "supports_pixel_map": True,
            "tested_runtime": "torch",
            "model_info_command": "pyimgano-benchmark --model-info vision_openclip_patch_map --json",
            "suggested_commands": [
                "pyimgano-benchmark --model-info vision_openclip_patch_map --json",
                "pyimgano-doctor --recommend-extras --for-command infer --json",
            ],
            "next_step_commands": [
                "pyimgano-doctor --recommend-extras --for-command infer --json",
            ],
        }
    )

    assert "- workflow_stage: discover" in lines
    assert "- supports_pixel_map: True" in lines
    assert "- tested_runtime: torch" in lines
    assert (
        "- model_info_command: pyimgano-benchmark --model-info vision_openclip_patch_map --json"
        in lines
    )


def test_format_extra_recommendation_lines_renders_runs_structured_commands() -> None:
    from pyimgano.doctor_rendering import format_extra_recommendation_lines

    lines = format_extra_recommendation_lines(
        {
            "target_kind": "command",
            "target": "runs",
            "quality_command": "pyimgano runs quality runs/<run_dir> --require-status audited --json",
            "acceptance_command": (
                "pyimgano runs acceptance runs/<run_dir> --require-status audited "
                "--check-bundle-hashes --json"
            ),
            "bundle_audit_command": (
                "pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json"
            ),
        }
    )

    assert (
        "- quality_command: pyimgano runs quality runs/<run_dir> --require-status audited --json"
        in lines
    )
    assert (
        "- acceptance_command: pyimgano runs acceptance runs/<run_dir> --require-status audited "
        "--check-bundle-hashes --json" in lines
    )
    assert (
        "- bundle_audit_command: pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle "
        "--check-hashes --json" in lines
    )
