from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_STARTER_BENCHMARK_NAME = "official_mvtec_industrial_v4_cpu_offline.json"


@dataclass(frozen=True)
class WorkflowStage:
    key: str
    title: str
    commands: tuple[str, ...]


@dataclass(frozen=True)
class CommandWorkflowGuidance:
    target_kind: str
    target: str
    workflow_stage: str
    recipe_list_command: str | None = None
    recipe_info_command: str | None = None
    dry_run_command: str | None = None
    recipe_run_command: str | None = None
    export_command: str | None = None
    infer_followup_command: str | None = None
    preset_infer_command: str | None = None
    from_run_infer_command: str | None = None
    quality_command: str | None = None
    acceptance_command: str | None = None
    bundle_audit_command: str | None = None
    suggested_commands: tuple[str, ...] = ()
    next_step_commands: tuple[str, ...] = ()
    artifact_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelWorkflowGuidance:
    target_kind: str
    target: str
    workflow_stage: str
    model_info_command: str
    suggested_commands: tuple[str, ...]
    next_step_commands: tuple[str, ...]


@dataclass(frozen=True)
class StarterBenchmarkGuidance:
    target_kind: str
    target: str
    list_command: str
    info_command: str
    run_command: str


@dataclass(frozen=True)
class StarterPathGuidance:
    name: str
    title: str
    summary: str
    commands: tuple[str, ...]


_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        key="discover",
        title="Discover",
        commands=(
            "pyim --list models --objective latency --selection-profile cpu-screening --topk 5",
        ),
    ),
    WorkflowStage(
        key="benchmark",
        title="Benchmark",
        commands=(
            "pyimgano doctor --recommend-extras --for-command benchmark --json",
            "pyimgano benchmark --list-starter-configs",
        ),
    ),
    WorkflowStage(
        key="train",
        title="Train",
        commands=(
            "pyimgano doctor --recommend-extras --for-command train --json",
            "pyimgano train --list-recipes",
            "pyimgano train --recipe-info industrial-adapt --json",
            "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
            "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
        ),
    ),
    WorkflowStage(
        key="export",
        title="Export",
        commands=(
            "pyimgano doctor --recommend-extras --for-command export-onnx --json",
            "pyimgano doctor --recommend-extras --for-command export-torchscript --json",
        ),
    ),
    WorkflowStage(
        key="infer",
        title="Infer",
        commands=(
            "pyimgano doctor --recommend-extras --for-command infer --json",
            "pyimgano-infer --from-run runs/<run_dir> --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
        ),
    ),
    WorkflowStage(
        key="validate",
        title="Validate",
        commands=(
            "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
        ),
    ),
    WorkflowStage(
        key="gate",
        title="Gate",
        commands=(
            "pyimgano doctor --recommend-extras --for-command runs --json",
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
        ),
    ),
)

_INDUSTRIAL_FAST_PATH_COMMANDS: tuple[str, ...] = (
    "pyimgano doctor --recommend-extras --for-command train --json",
    "pyimgano doctor --recommend-extras --for-command infer --json",
    "pyimgano doctor --recommend-extras --for-command runs --json",
    "pyimgano train --list-recipes",
    "pyimgano train --recipe-info industrial-adapt --json",
    "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
    "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
    "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
    "pyimgano bundle run runs/<run_dir>/deploy_bundle --image-dir /path/to/images --output-dir ./bundle_run --json",
    "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
    "pyimgano runs quality runs/<run_dir> --require-status audited --json",
    "pyimgano runs acceptance runs/<run_dir> --require-status audited --json",
)

_BENCHMARK_PUBLICATION_COMMANDS: tuple[str, ...] = (
    "pyimgano benchmark --list-starter-configs",
    "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
    "pyimgano benchmark --list-official-configs",
    "pyimgano benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
    "pyimgano runs acceptance /path/to/suite_export --json",
    "pyimgano runs publication /path/to/suite_export --json",
)

_ARTIFACT_ACCEPTANCE_COMMANDS: tuple[str, ...] = (
    "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
    "pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json",
)

_FIRST_TEN_MINUTES_COMMANDS: tuple[str, ...] = (
    "pyimgano-doctor --profile first-run --json",
    "pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained",
    "pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json",
    "pyimgano-benchmark --dataset custom --root ./_demo_custom_dataset --suite industrial-ci --resize 32 32 --limit-train 2 --limit-test 2 --no-pretrained --save-run --output-dir ./_demo_benchmark_run --suite-export csv",
    "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir ./_demo_custom_dataset/train/normal --input ./_demo_custom_dataset/test --save-jsonl ./_demo_results.jsonl",
    "pyimgano runs quality ./_demo_benchmark_run --json",
)

_DEPLOY_SMOKE_COMMANDS: tuple[str, ...] = (
    "pyimgano-doctor --profile deploy-smoke --json",
    "pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained",
    "pyimgano-train --config examples/configs/deploy_smoke_custom_cpu.json --root ./_demo_custom_dataset --export-infer-config --export-deploy-bundle",
    "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
    "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
    "pyimgano runs quality runs/<run_dir> --json",
)

_STARTER_PATHS: tuple[StarterPathGuidance, ...] = (
    StarterPathGuidance(
        name="deploy-smoke",
        title="Deployment Smoke Path",
        summary="Smallest offline-safe path from demo dataset creation to deploy bundle validation.",
        commands=_DEPLOY_SMOKE_COMMANDS,
    ),
    StarterPathGuidance(
        name="first-run",
        title="First 10 Minutes",
        summary="Smallest offline-safe operator path from environment check to benchmarked inference artifacts.",
        commands=_FIRST_TEN_MINUTES_COMMANDS,
    ),
    StarterPathGuidance(
        name="benchmark",
        title="First Benchmark Run",
        summary="Starter benchmark discovery plus one publication-aware benchmark loop.",
        commands=(
            "pyimgano-doctor --profile benchmark --dataset-target /path/to/dataset_root --json",
            "pyimgano benchmark --list-starter-configs",
            "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
            "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json",
            "pyimgano runs publication /path/to/suite_export --json",
        ),
    ),
    StarterPathGuidance(
        name="deploy",
        title="Deployment Check",
        summary="Validate deploy artifacts and acceptance signals before shipping a run or bundle.",
        commands=(
            "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json",
            "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
            "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
        ),
    ),
    StarterPathGuidance(
        name="publish",
        title="Publication Gate",
        summary="Gate a suite export with the same publication trust checks used by release workflows.",
        commands=(
            "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json",
            "pyimgano runs acceptance /path/to/suite_export --json",
            "pyimgano runs publication /path/to/suite_export --json",
        ),
    ),
)

_COMMAND_WORKFLOW_GUIDANCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "benchmark": CommandWorkflowGuidance(
        target_kind="command",
        target="benchmark",
        workflow_stage="benchmark",
        suggested_commands=(
            "pyimgano benchmark --list-starter-configs",
            "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
            "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json",
        ),
        next_step_commands=("pyimgano-doctor --recommend-extras --for-command train --json",),
        artifact_hints=(
            "leaderboard.csv",
            "best_by_baseline.csv",
            "skipped.csv",
            "leaderboard_metadata.json",
        ),
    ),
    "train": CommandWorkflowGuidance(
        target_kind="command",
        target="train",
        workflow_stage="train",
        recipe_list_command="pyimgano train --list-recipes",
        recipe_info_command="pyimgano train --recipe-info industrial-adapt --json",
        dry_run_command="pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
        recipe_run_command=(
            "pyimgano train --config examples/configs/industrial_adapt_audited.json "
            "--export-infer-config --export-deploy-bundle"
        ),
        suggested_commands=(
            "pyimgano train --list-recipes",
            "pyimgano train --recipe-info industrial-adapt --json",
            "pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json",
            "pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle",
            "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
        ),
        next_step_commands=("pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",),
        artifact_hints=(
            "artifacts/infer_config.json",
            "artifacts/calibration_card.json",
            "deploy_bundle/infer_config.json",
            "deploy_bundle/bundle_manifest.json",
        ),
    ),
    "export-onnx": CommandWorkflowGuidance(
        target_kind="command",
        target="export-onnx",
        workflow_stage="validate",
        export_command="pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained",
        infer_followup_command="pyimgano-doctor --recommend-extras --for-command infer --json",
        suggested_commands=(
            "pyimgano-export-onnx --backbone resnet18 --output /tmp/embed.onnx --no-pretrained",
            "pyimgano-doctor --recommend-extras --for-command infer --json",
        ),
        next_step_commands=("pyimgano-doctor --recommend-extras --for-command infer --json",),
        artifact_hints=(
            "embed.onnx",
            "onnx sweep JSON (optional)",
        ),
    ),
    "export-torchscript": CommandWorkflowGuidance(
        target_kind="command",
        target="export-torchscript",
        workflow_stage="validate",
        export_command="pyimgano-export-torchscript --backbone resnet18 --output /tmp/embed.ts --no-pretrained",
        infer_followup_command="pyimgano-doctor --recommend-extras --for-command infer --json",
        suggested_commands=(
            "pyimgano-export-torchscript --backbone resnet18 --output /tmp/embed.ts --no-pretrained",
            "pyimgano-doctor --recommend-extras --for-command infer --json",
        ),
        next_step_commands=("pyimgano-doctor --recommend-extras --for-command infer --json",),
        artifact_hints=("embed.ts",),
    ),
    "infer": CommandWorkflowGuidance(
        target_kind="command",
        target="infer",
        workflow_stage="infer",
        preset_infer_command=(
            "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir "
            "/path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl"
        ),
        from_run_infer_command=(
            "pyimgano-infer --from-run runs/<run_dir> --input /path/to/images "
            "--save-jsonl /tmp/pyimgano_results.jsonl"
        ),
        suggested_commands=(
            "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir /path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
            "pyimgano-infer --from-run runs/<run_dir> --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
        ),
        next_step_commands=("pyimgano-doctor --recommend-extras --for-command runs --json",),
        artifact_hints=(
            "results.jsonl",
            "masks/ (optional)",
            "overlays/ (optional)",
            "regions.jsonl (optional)",
        ),
    ),
    "runs": CommandWorkflowGuidance(
        target_kind="command",
        target="runs",
        workflow_stage="gate",
        quality_command="pyimgano runs quality runs/<run_dir> --require-status audited --json",
        acceptance_command=(
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json"
        ),
        bundle_audit_command=(
            "pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json"
        ),
        suggested_commands=(
            "pyimgano runs quality runs/<run_dir> --require-status audited --json",
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
        ),
        next_step_commands=("pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json",),
        artifact_hints=(
            "report.json",
            "config.json",
            "environment.json",
            "leaderboard_metadata.json (suite exports)",
        ),
    ),
    "demo": CommandWorkflowGuidance(target_kind="command", target="demo", workflow_stage="discover"),
}

_MODEL_WORKFLOW_STAGE = "discover"


def list_workflow_stages() -> list[WorkflowStage]:
    return list(_WORKFLOW_STAGES)


def default_starter_benchmark_name() -> str:
    return _DEFAULT_STARTER_BENCHMARK_NAME


def starter_benchmark_list_command() -> str:
    return "pyimgano benchmark --list-starter-configs"


def starter_benchmark_info_command(name: str | None = None) -> str:
    target = str(name or _DEFAULT_STARTER_BENCHMARK_NAME)
    return f"pyimgano benchmark --starter-config-info {target} --json"


def starter_benchmark_run_command(name: str | None = None) -> str:
    target = str(name or _DEFAULT_STARTER_BENCHMARK_NAME)
    return f"pyimgano-benchmark --config {target}"


def starter_benchmark_guidance(name: str | None = None) -> StarterBenchmarkGuidance:
    target = str(name or _DEFAULT_STARTER_BENCHMARK_NAME)
    return StarterBenchmarkGuidance(
        target_kind="starter-benchmark",
        target=target,
        list_command=starter_benchmark_list_command(),
        info_command=starter_benchmark_info_command(target),
        run_command=starter_benchmark_run_command(target),
    )


def industrial_fast_path_commands() -> list[str]:
    return list(_INDUSTRIAL_FAST_PATH_COMMANDS)


def benchmark_publication_commands() -> list[str]:
    return list(_BENCHMARK_PUBLICATION_COMMANDS)


def artifact_acceptance_commands() -> list[str]:
    return list(_ARTIFACT_ACCEPTANCE_COMMANDS)


def first_ten_minutes_commands() -> list[str]:
    return list(_FIRST_TEN_MINUTES_COMMANDS)


def deploy_smoke_commands() -> list[str]:
    return list(_DEPLOY_SMOKE_COMMANDS)


def list_starter_paths() -> list[StarterPathGuidance]:
    return list(_STARTER_PATHS)


def starter_path_by_name(name: str) -> StarterPathGuidance | None:
    key = str(name).strip()
    for path in _STARTER_PATHS:
        if path.name == key:
            return path
    return None


def workflow_stage_for_command(command_name: str) -> str | None:
    info = _COMMAND_WORKFLOW_GUIDANCE.get(str(command_name).strip())
    if info is None:
        return None
    return str(info.workflow_stage)


def command_workflow_guidance(command_name: str) -> CommandWorkflowGuidance | None:
    return _COMMAND_WORKFLOW_GUIDANCE.get(str(command_name).strip())


def next_step_commands_for_command(command_name: str) -> list[str]:
    info = _COMMAND_WORKFLOW_GUIDANCE.get(str(command_name).strip())
    if info is None:
        return []
    return [str(item) for item in info.next_step_commands]


def suggested_commands_for_command(command_name: str) -> list[str]:
    info = _COMMAND_WORKFLOW_GUIDANCE.get(str(command_name).strip())
    if info is None:
        return []
    return [str(item) for item in info.suggested_commands]


def artifact_hints_for_command(command_name: str) -> list[str]:
    info = _COMMAND_WORKFLOW_GUIDANCE.get(str(command_name).strip())
    if info is None:
        return []
    return [str(item) for item in info.artifact_hints]


def workflow_stage_for_model(_model_name: str) -> str:
    return _MODEL_WORKFLOW_STAGE


def model_info_command_for_model(model_name: str) -> str:
    return f"pyimgano-benchmark --model-info {model_name} --json"


def suggested_commands_for_model(model_name: str) -> list[str]:
    return [
        model_info_command_for_model(model_name),
        "pyimgano-doctor --recommend-extras --for-command infer --json",
    ]


def next_step_commands_for_model(_model_name: str) -> list[str]:
    return ["pyimgano-doctor --recommend-extras --for-command infer --json"]


def model_workflow_guidance(model_name: str) -> ModelWorkflowGuidance:
    return ModelWorkflowGuidance(
        target_kind="model",
        target=str(model_name),
        workflow_stage=workflow_stage_for_model(model_name),
        model_info_command=model_info_command_for_model(model_name),
        suggested_commands=tuple(suggested_commands_for_model(model_name)),
        next_step_commands=tuple(next_step_commands_for_model(model_name)),
    )


__all__ = [
    "CommandWorkflowGuidance",
    "ModelWorkflowGuidance",
    "StarterBenchmarkGuidance",
    "StarterPathGuidance",
    "WorkflowStage",
    "artifact_hints_for_command",
    "artifact_acceptance_commands",
    "benchmark_publication_commands",
    "command_workflow_guidance",
    "default_starter_benchmark_name",
    "deploy_smoke_commands",
    "first_ten_minutes_commands",
    "industrial_fast_path_commands",
    "list_starter_paths",
    "list_workflow_stages",
    "model_workflow_guidance",
    "model_info_command_for_model",
    "next_step_commands_for_command",
    "next_step_commands_for_model",
    "suggested_commands_for_command",
    "suggested_commands_for_model",
    "starter_benchmark_guidance",
    "starter_benchmark_info_command",
    "starter_benchmark_list_command",
    "starter_benchmark_run_command",
    "starter_path_by_name",
    "workflow_stage_for_command",
    "workflow_stage_for_model",
]
