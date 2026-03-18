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
        payload = dict(validation.payload)
        payload["validation_trust"] = dict(validation.trust_summary)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    model = validation.payload.get("model", {}) or {}
    model_name = model.get("name", None)
    category = validation.payload.get("category", None)
    artifact_quality = validation.payload.get("artifact_quality", {}) or {}
    trust_summary = dict(validation.trust_summary)
    ckpt = validation.resolved_checkpoint_path
    model_ckpt = validation.resolved_model_checkpoint_path
    msg = "ok"
    if model_name is not None:
        msg += f": model={model_name}"
    if category is not None:
        msg += f" category={category}"
    if artifact_quality:
        status = artifact_quality.get("status", None)
        threshold_scope = artifact_quality.get("threshold_scope", None)
        if status is not None:
            msg += f" audit_status={status}"
        if threshold_scope is not None:
            msg += f" threshold_scope={threshold_scope}"
        if "has_deploy_bundle" in artifact_quality:
            msg += f" deploy_bundle={str(bool(artifact_quality['has_deploy_bundle'])).lower()}"
        if "has_bundle_manifest" in artifact_quality:
            msg += (
                f" bundle_manifest={str(bool(artifact_quality['has_bundle_manifest'])).lower()}"
            )
        if "required_bundle_artifacts_present" in artifact_quality:
            msg += (
                " bundle_required="
                f"{str(bool(artifact_quality['required_bundle_artifacts_present'])).lower()}"
            )
    trust_status = trust_summary.get("status", None)
    if trust_status is not None:
        msg += f" trust_status={trust_status}"
    if ckpt is not None:
        msg += f" checkpoint={ckpt}"
    if model_ckpt is not None:
        msg += f" model_checkpoint={model_ckpt}"
    print(msg)
    trust_signals = trust_summary.get("trust_signals", {})
    if isinstance(trust_signals, dict):
        for key, value in trust_signals.items():
            print(f"trust_signal.{key}={value}")
    for item in trust_summary.get("degraded_by", []):
        print(f"degraded_by={item}")
    audit_refs = trust_summary.get("audit_refs", {})
    if isinstance(audit_refs, dict):
        for key, value in audit_refs.items():
            print(f"audit_ref.{key}={value}")
    return 0
