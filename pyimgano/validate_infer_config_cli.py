from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python pyimgano/validate_infer_config_cli.py`, Python sets sys.path[0]
    # to `pyimgano/` rather than the repo root. Add the repo root so `import pyimgano` works
    # without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-validate-infer-config",
        description="Validate a pyimgano workbench infer-config.json for deploy-style inference.",
    )
    parser.add_argument(
        "infer_config",
        help="Path to infer_config.json (exported by pyimgano-train --export-infer-config)",
    )
    parser.add_argument(
        "--infer-category",
        default=None,
        help="Optional category selection when infer-config contains multiple categories",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip file existence checks (e.g. checkpoints) for portability",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the normalized infer-config payload as JSON to stdout",
    )
    return parser


def _emit_json_validation_payload(validation: object) -> int:
    payload = dict(validation.payload)
    payload["validation_trust"] = dict(validation.trust_summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _artifact_quality_segments(artifact_quality: dict[str, object]) -> list[str]:
    segments: list[str] = []
    status = artifact_quality.get("status", None)
    threshold_scope = artifact_quality.get("threshold_scope", None)
    if status is not None:
        segments.append(f"audit_status={status}")
    if threshold_scope is not None:
        segments.append(f"threshold_scope={threshold_scope}")
    if "has_deploy_bundle" in artifact_quality:
        segments.append(
            f"deploy_bundle={str(bool(artifact_quality['has_deploy_bundle'])).lower()}"
        )
    if "has_bundle_manifest" in artifact_quality:
        segments.append(
            f"bundle_manifest={str(bool(artifact_quality['has_bundle_manifest'])).lower()}"
        )
    if "required_bundle_artifacts_present" in artifact_quality:
        segments.append(
            "bundle_required="
            f"{str(bool(artifact_quality['required_bundle_artifacts_present'])).lower()}"
        )
    return segments


def _plain_validation_summary(validation: object) -> str:
    model = validation.payload.get("model", {}) or {}
    model_name = model.get("name", None)
    category = validation.payload.get("category", None)
    artifact_quality = validation.payload.get("artifact_quality", {}) or {}
    trust_summary = dict(validation.trust_summary)
    segments = ["ok"]
    if model_name is not None:
        segments.append(f"model={model_name}")
    if category is not None:
        segments.append(f"category={category}")
    if artifact_quality:
        segments.extend(_artifact_quality_segments(artifact_quality))
    trust_status = trust_summary.get("status", None)
    if trust_status is not None:
        segments.append(f"trust_status={trust_status}")
    if validation.resolved_checkpoint_path is not None:
        segments.append(f"checkpoint={validation.resolved_checkpoint_path}")
    if validation.resolved_model_checkpoint_path is not None:
        segments.append(f"model_checkpoint={validation.resolved_model_checkpoint_path}")
    return ": ".join(segments[:1]) + (f" {' '.join(segments[1:])}" if len(segments) > 1 else "")


def _print_prefixed_mapping(prefix: str, value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        print(f"{prefix}.{key}={item}")


def _emit_plain_validation_output(validation: object) -> int:
    trust_summary = dict(validation.trust_summary)
    print(_plain_validation_summary(validation))
    _print_prefixed_mapping("trust_signal", trust_summary.get("trust_signals", {}))
    for item in trust_summary.get("degraded_by", []):
        print(f"degraded_by={item}")
    _print_prefixed_mapping("audit_ref", trust_summary.get("audit_refs", {}))
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_root_on_sys_path()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from pyimgano.inference.validate_infer_config import validate_infer_config_file

        validation = validate_infer_config_file(
            Path(str(args.infer_config)),
            category=(str(args.infer_category) if args.infer_category is not None else None),
            check_files=(not bool(args.no_check_files)),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(str(exc), file=sys.stderr)
        return 1

    for w in validation.warnings:
        print(f"warning: {w}", file=sys.stderr)

    if bool(args.json):
        return _emit_json_validation_payload(validation)
    return _emit_plain_validation_output(validation)
