"""Top-level convenience CLI for pyimgano."""

from __future__ import annotations

import importlib
import sys
from textwrap import dedent


_COMMANDS: dict[str, tuple[str, str]] = {
    "benchmark": ("pyimgano.cli", "Benchmarking, suites, and benchmark discovery."),
    "infer": ("pyimgano.infer_cli", "Inference, deploy bundles, and model discovery."),
    "train": ("pyimgano.train_cli", "Workbench training and artifact export."),
    "runs": ("pyimgano.runs_cli", "Run indexing, comparison, and quality gates."),
    "weights": ("pyimgano.weights_cli", "Weights manifest/model card tooling."),
    "datasets": ("pyimgano.datasets_cli", "Dataset inspection and utilities."),
    "features": ("pyimgano.feature_cli", "Feature extraction utilities."),
    "defects": ("pyimgano.defects_cli", "Defects extraction utilities."),
    "manifest": ("pyimgano.manifest_cli", "Manifest generation and validation."),
    "doctor": ("pyimgano.doctor_cli", "Environment and accelerator diagnostics."),
    "synthesize": ("pyimgano.synthesize_cli", "Synthetic defect data generation."),
    "demo": ("pyimgano.demo_cli", "Demo and one-click baseline workflows."),
    "export-onnx": ("pyimgano.onnx_export_cli", "ONNX export helpers."),
    "export-torchscript": ("pyimgano.torchscript_export_cli", "TorchScript export helpers."),
    "validate-infer-config": (
        "pyimgano.validate_infer_config_cli",
        "Infer-config validation.",
    ),
    "robust-benchmark": (
        "pyimgano.robust_cli",
        "Robustness benchmarking and corruption summaries.",
    ),
}


def _help_text() -> str:
    command_lines = "\n".join(
        f"  {name:<22} {description}" for name, (_module, description) in _COMMANDS.items()
    )
    return dedent(
        f"""\
        usage:
          pyimgano --list [KIND] [filters...]
          pyimgano list [KIND] [filters...]
          pyimgano -- list [KIND] [filters...]
          pyimgano <command> [args...]
          pyimgano help [command]

        discovery shortcuts:
          --list [KIND]            Same discovery surface as `pyim --list`.
          --audit-metadata         Audit model metadata contract.
          --json                   JSON output for discovery commands.

        commands:
        {command_lines}

        industrial fast-path:
          pyimgano train --config examples/configs/industrial_adapt_audited.json --export-infer-config --export-deploy-bundle
          pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json
          pyimgano runs quality runs/<run_dir> --require-status audited --json
          pyimgano runs acceptance runs/<run_dir> --require-status audited --json

        benchmark publication:
          pyimgano benchmark --list-official-configs
          pyimgano benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json
          pyimgano runs acceptance /path/to/suite_export --json
          pyimgano runs publication /path/to/suite_export --json

        artifact acceptance:
          pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json
          pyimgano weights audit-bundle runs/<run_dir>/deploy_bundle --check-hashes --json

        examples:
          pyimgano --list
          pyimgano list models
          pyimgano -- list models --json
          pyimgano --list models --family patchcore
          pyimgano train --help
          pyimgano runs quality /path/to/run --json

        notes:
          - `pyim` remains available as the short discovery alias.
          - `pyimgano list ...` and `pyimgano -- list ...` are aliases for `pyimgano --list ...`.
          - Existing `pyimgano-*` entry points remain supported.
        """
    )


def _print_help() -> None:
    print(_help_text())


def _load_command_main(module_path: str):
    module = importlib.import_module(module_path)
    return getattr(module, "main")


def _dispatch_command(name: str, argv: list[str]) -> int:
    module_path, _description = _COMMANDS[name]
    return int(_load_command_main(module_path)(argv))


def _run_discovery_cli(argv: list[str]) -> int:
    from pyimgano.pyim_cli import main as pyim_main

    return int(pyim_main(argv))


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _normalize_root_args(args: list[str]) -> list[str]:
    normalized = list(args)
    if not normalized:
        return normalized

    if normalized[0] == "--":
        if len(normalized) == 1:
            return ["--help"]
        normalized = normalized[1:]

    if not normalized:
        return ["--help"]

    first = str(normalized[0]).strip()
    if first == "list":
        rest = normalized[1:]
        if not rest:
            return ["--list"]
        return ["--list", *rest]
    if first == "audit-metadata":
        return ["--audit-metadata", *normalized[1:]]
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = _normalize_root_args(list(sys.argv[1:] if argv is None else argv))

    if not args:
        _print_help()
        return 0

    first = str(args[0]).strip()

    if first in {"-h", "--help"}:
        _print_help()
        return 0

    try:
        if first == "help":
            if len(args) == 1:
                _print_help()
                return 0
            target = str(args[1]).strip()
            if target not in _COMMANDS:
                print(f"Unknown command: {target}", file=sys.stderr)
                return 2
            return _dispatch_command(target, ["--help"])

        if first.startswith("-"):
            return _run_discovery_cli(args)

        if first not in _COMMANDS:
            print(f"Unknown command: {first}", file=sys.stderr)
            print("Run `pyimgano --help` to see available commands.", file=sys.stderr)
            return 2

        return _dispatch_command(first, args[1:])
    except SystemExit as exc:
        return _system_exit_code(exc)


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
