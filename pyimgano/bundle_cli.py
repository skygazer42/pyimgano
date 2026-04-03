from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyimgano.bundle_rendering as bundle_rendering
import pyimgano.cli_output as cli_output
import pyimgano.services.bundle_watch_service as bundle_watch_service
from pyimgano.bundle_cli_helpers import (
    build_batch_gate_summary as _build_batch_gate_summary_helper,
)
from pyimgano.bundle_cli_helpers import (
    build_input_source_summary as _build_input_source_summary_helper,
)
from pyimgano.bundle_cli_helpers import (
    build_reason_codes as _build_reason_codes_helper,
)
from pyimgano.bundle_cli_helpers import (
    run_exit_code as _run_exit_code_helper,
)
from pyimgano.bundle_cli_helpers import (
    validate_exit_code as _validate_exit_code_helper,
)
from pyimgano.datasets.manifest_tools import iter_manifest_rows
from pyimgano.infer_cli_inputs import collect_image_paths
from pyimgano.inference.validate_infer_config import validate_infer_config_file
from pyimgano.reporting.deploy_bundle import (
    normalize_deploy_bundle_runtime_policy,
    validate_deploy_bundle_manifest,
    validate_deploy_bundle_handoff_report,
)
from pyimgano.services.bundle_run_service import BundleInferenceBatchRequest
from pyimgano.services.bundle_run_service import run_bundle_inference_batch
from pyimgano.utils.security import FileHasher
from pyimgano.weights.bundle_audit import evaluate_bundle_weights_audit

_BUNDLE_VALIDATE_SCHEMA_VERSION = 1
_BUNDLE_RUN_REPORT_SCHEMA_VERSION = 1
_VALIDATE_REASON_CODE_MAP = {
    "bundle_not_found": "BUNDLE_NOT_FOUND",
    "bundle_not_directory": "BUNDLE_NOT_DIRECTORY",
    "missing_infer_config": "BUNDLE_MISSING_INFER_CONFIG",
    "invalid_infer_config": "BUNDLE_INVALID_INFER_CONFIG",
    "missing_manifest": "BUNDLE_MISSING_MANIFEST",
    "invalid_manifest": "BUNDLE_INVALID_MANIFEST",
    "invalid_handoff_report": "BUNDLE_INVALID_HANDOFF_REPORT",
    "required_artifacts_missing": "BUNDLE_REQUIRED_ARTIFACTS_MISSING",
    "bundle_weights_not_ready": "BUNDLE_WEIGHTS_NOT_READY",
}
_RUN_REASON_CODE_MAP = {
    "invalid_input_manifest": "RUN_INPUT_MANIFEST_INVALID",
    "input_not_found": "RUN_INPUT_NOT_FOUND",
    "no_input_images": "RUN_NO_INPUT_IMAGES",
    "pixel_outputs_not_supported": "RUN_PIXEL_OUTPUTS_NOT_SUPPORTED",
    "inference_failed": "RUN_INFERENCE_FAILED",
}
_RUN_BATCH_GATE_REASON_CODE_MAP = {
    "max_anomaly_rate": "RUN_BATCH_ANOMALY_RATE_EXCEEDED",
    "max_reject_rate": "RUN_BATCH_REJECT_RATE_EXCEEDED",
    "max_error_rate": "RUN_BATCH_ERROR_RATE_EXCEEDED",
    "min_processed": "RUN_BATCH_MIN_PROCESSED_NOT_MET",
}
_BATCH_GATE_KEYS = (
    "max_anomaly_rate",
    "max_reject_rate",
    "max_error_rate",
    "min_processed",
)


def _parse_rate_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float between 0 and 1.") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be a float between 0 and 1.")
    return float(value)


def _parse_min_processed_arg(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to 1.") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to 1.")
    return int(value)


def _parse_nonnegative_float_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float greater than or equal to 0.") from exc
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be a float greater than or equal to 0.")
    return float(value)


def _parse_webhook_header_arg(text: str) -> tuple[str, str]:
    key, sep, value = str(text).partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("--webhook-header must use KEY=VALUE syntax.")
    key = key.strip()
    value = value.strip()
    if not key:
        raise argparse.ArgumentTypeError("--webhook-header key must be non-empty.")
    return key, value


def _resolve_secret_from_cli_or_env(
    *,
    value: str | None,
    env_var: str | None,
    option_name: str,
    env_option_name: str,
) -> str | None:
    if value is not None:
        return str(value)
    if env_var is None:
        return None
    raw = os.environ.get(str(env_var))
    if raw is None or not str(raw).strip():
        raise ValueError(
            f"{env_option_name} requires a non-empty environment variable: {env_var}"
        )
    return str(raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-bundle",
        description="Validate and execute a CPU offline QC deploy bundle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate deploy bundle artifacts and machine-readable contract fields.",
    )
    validate_parser.add_argument("bundle_dir", help="Path to deploy bundle directory.")
    validate_parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Verify recorded bundle hashes when bundle metadata provides them.",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON payload to stdout.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run offline inference from a validated deploy bundle with fixed output contracts.",
    )
    run_parser.add_argument("bundle_dir", help="Path to deploy bundle directory.")
    source = run_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image-dir",
        default=None,
        help="Directory of input images. Scanned recursively for supported image types.",
    )
    source.add_argument(
        "--image",
        default=None,
        help="Path to a single image input.",
    )
    source.add_argument(
        "--input-manifest",
        default=None,
        help="JSONL input manifest with at least image_path and category fields.",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Writes results.jsonl and run_report.json here.",
    )
    run_parser.add_argument(
        "--max-anomaly-rate",
        type=_parse_rate_arg,
        default=None,
        help="Block the batch if anomalous / processed exceeds this rate (0-1).",
    )
    run_parser.add_argument(
        "--max-reject-rate",
        type=_parse_rate_arg,
        default=None,
        help="Block the batch if rejected / processed exceeds this rate (0-1).",
    )
    run_parser.add_argument(
        "--max-error-rate",
        type=_parse_rate_arg,
        default=None,
        help="Block the batch if error / processed exceeds this rate (0-1).",
    )
    run_parser.add_argument(
        "--min-processed",
        type=_parse_min_processed_arg,
        default=None,
        help="Block the batch if fewer than this many records are processed.",
    )
    run_parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Verify recorded bundle hashes before executing inference.",
    )
    run_parser.add_argument(
        "--export-masks",
        action="store_true",
        help="Write defect masks to <output-dir>/masks when the bundle supports pixel outputs.",
    )
    run_parser.add_argument(
        "--export-overlays",
        action="store_true",
        help=(
            "Write inspection overlays to <output-dir>/overlays when the bundle supports pixel outputs."
        ),
    )
    run_parser.add_argument(
        "--export-defects-regions",
        action="store_true",
        help=(
            "Write defect regions JSONL to <output-dir>/defects_regions.jsonl when the bundle "
            "supports pixel outputs."
        ),
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit run_report.json payload to stdout after writing it.",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Poll a hot folder and append stable inputs to deploy-bundle inference outputs.",
    )
    watch_parser.add_argument("bundle_dir", help="Path to deploy bundle directory.")
    watch_parser.add_argument(
        "--watch-dir",
        required=True,
        help="Directory to scan recursively for image inputs.",
    )
    watch_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Writes aggregate watch artifacts here.",
    )
    watch_parser.add_argument(
        "--poll-seconds",
        type=_parse_nonnegative_float_arg,
        default=1.0,
        help="Seconds to sleep between scans in long-running mode. Default: 1.0.",
    )
    watch_parser.add_argument(
        "--settle-seconds",
        type=_parse_nonnegative_float_arg,
        default=2.0,
        help="Minimum stable age before a file is processed. Default: 2.0.",
    )
    watch_parser.add_argument(
        "--once",
        action="store_true",
        help="Process only the current stable backlog, then exit.",
    )
    watch_parser.add_argument(
        "--state-file",
        default=None,
        help="Optional override for watch state path. Default: <output-dir>/watch_state.json",
    )
    watch_parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional callback URL. When set, each processed watch record is POSTed as JSON.",
    )
    watch_parser.add_argument(
        "--webhook-bearer-token",
        default=None,
        help="Optional bearer token added as Authorization: Bearer <token> for webhook delivery.",
    )
    watch_parser.add_argument(
        "--webhook-bearer-token-env",
        default=None,
        help="Environment variable name used to resolve the webhook bearer token.",
    )
    watch_parser.add_argument(
        "--webhook-signing-secret",
        default=None,
        help="Optional HMAC secret used to emit signed webhook headers.",
    )
    watch_parser.add_argument(
        "--webhook-signing-secret-env",
        default=None,
        help="Environment variable name used to resolve the webhook signing secret.",
    )
    watch_parser.add_argument(
        "--webhook-header",
        action="append",
        default=None,
        type=_parse_webhook_header_arg,
        metavar="KEY=VALUE",
        help="Repeatable custom header for webhook delivery.",
    )
    watch_parser.add_argument(
        "--webhook-timeout-seconds",
        type=_parse_nonnegative_float_arg,
        default=5.0,
        help="Timeout for webhook POST requests. Default: 5.0.",
    )
    watch_parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Verify recorded bundle hashes before polling.",
    )
    watch_parser.add_argument(
        "--max-anomaly-rate",
        type=_parse_rate_arg,
        default=None,
        help="Block the current watch cycle if anomalous / processed exceeds this rate (0-1).",
    )
    watch_parser.add_argument(
        "--max-reject-rate",
        type=_parse_rate_arg,
        default=None,
        help="Block the current watch cycle if rejected / processed exceeds this rate (0-1).",
    )
    watch_parser.add_argument(
        "--max-error-rate",
        type=_parse_rate_arg,
        default=None,
        help="Block the current watch cycle if error / processed exceeds this rate (0-1).",
    )
    watch_parser.add_argument(
        "--min-processed",
        type=_parse_min_processed_arg,
        default=None,
        help="Block the current watch cycle if fewer than this many records are processed.",
    )
    watch_parser.add_argument(
        "--export-masks",
        action="store_true",
        help="Write defect masks to <output-dir>/masks when the bundle supports pixel outputs.",
    )
    watch_parser.add_argument(
        "--export-overlays",
        action="store_true",
        help="Write inspection overlays to <output-dir>/overlays when the bundle supports pixel outputs.",
    )
    watch_parser.add_argument(
        "--export-defects-regions",
        action="store_true",
        help="Write aggregate defect regions JSONL to <output-dir>/defects_regions.jsonl.",
    )
    watch_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit watch_report.json payload to stdout after the current cycle or `--once` run.",
    )
    return parser


def _infer_config_validation_payload(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": False,
        "valid": None,
        "warnings": [],
        "errors": [],
        "trust_summary": {},
        "resolved_checkpoint_path": None,
        "resolved_model_checkpoint_path": None,
    }


def _manifest_validation_payload(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": False,
        "valid": None,
        "errors": [],
        "warnings": [],
        "payload": None,
    }


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object/dict.")
    return dict(payload)


def _evaluate_infer_config(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "infer_config.json"
    payload = _infer_config_validation_payload(path)
    if not path.is_file():
        payload["errors"] = ["Missing infer_config.json."]
        return payload

    payload["present"] = True
    try:
        validation = validate_infer_config_file(path, check_files=True)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        payload["valid"] = False
        payload["errors"] = [str(exc)]
        return payload

    payload["valid"] = True
    payload["warnings"] = list(validation.warnings)
    payload["trust_summary"] = dict(validation.trust_summary)
    payload["resolved_checkpoint_path"] = (
        str(validation.resolved_checkpoint_path)
        if validation.resolved_checkpoint_path is not None
        else None
    )
    payload["resolved_model_checkpoint_path"] = (
        str(validation.resolved_model_checkpoint_path)
        if validation.resolved_model_checkpoint_path is not None
        else None
    )
    return payload


def _evaluate_bundle_manifest(bundle_root: Path, *, check_hashes: bool) -> dict[str, Any]:
    path = bundle_root / "bundle_manifest.json"
    payload = _manifest_validation_payload(path)
    if not path.is_file():
        payload["errors"] = ["Missing bundle_manifest.json."]
        return payload

    payload["present"] = True
    try:
        manifest = _load_json_dict(path)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        payload["valid"] = False
        payload["errors"] = [str(exc)]
        return payload

    payload["payload"] = manifest
    payload["errors"] = validate_deploy_bundle_manifest(
        manifest,
        bundle_dir=bundle_root,
        check_hashes=bool(check_hashes),
    )
    payload["valid"] = len(payload["errors"]) == 0
    return payload


def _bundle_weights_payload(bundle_root: Path, *, check_hashes: bool) -> dict[str, Any]:
    metadata_present = any(
        (bundle_root / name).is_file() for name in ("model_card.json", "weights_manifest.json")
    )
    if not metadata_present:
        return {
            "applicable": False,
            "bundle_dir": str(bundle_root),
            "present": False,
            "valid": None,
            "ready": None,
            "status": "not_applicable",
            "missing_required": [],
            "warnings": [],
            "errors": [],
            "trust_summary": {},
        }

    return {
        "applicable": True,
        **evaluate_bundle_weights_audit(bundle_root, check_hashes=bool(check_hashes)),
    }


def _reason_codes(blocking_reasons: list[str], *, mapping: Mapping[str, str]) -> list[str]:
    return _build_reason_codes_helper(blocking_reasons, mapping=mapping)


def _validate_exit_code(payload: Mapping[str, Any]) -> int:
    return _validate_exit_code_helper(payload)


def _run_exit_code(status: str) -> int:
    return _run_exit_code_helper(status)


def _default_bundle_weights_payload(bundle_root: Path) -> dict[str, Any]:
    return {
        "applicable": False,
        "bundle_dir": str(bundle_root),
        "present": False,
        "valid": None,
        "ready": None,
        "status": "not_applicable",
        "missing_required": [],
        "warnings": [],
        "errors": [],
        "trust_summary": {},
    }


def _evaluate_bundle_artifacts(
    bundle_root: Path,
    *,
    check_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not bundle_root.is_dir():
        return {}, {}, _default_bundle_weights_payload(bundle_root), {
            "path": str(bundle_root / "handoff_report.json"),
            "present": False,
            "valid": None,
            "errors": [],
            "status": "not_applicable",
        }
    infer_config = _evaluate_infer_config(bundle_root)
    bundle_manifest = _evaluate_bundle_manifest(bundle_root, check_hashes=bool(check_hashes))
    bundle_weights = _bundle_weights_payload(bundle_root, check_hashes=bool(check_hashes))
    handoff_report = _evaluate_handoff_report(bundle_root)
    return infer_config, bundle_manifest, bundle_weights, handoff_report


def _evaluate_handoff_report(bundle_root: Path) -> dict[str, Any]:
    payload = {
        "path": str(bundle_root / "handoff_report.json"),
        "present": False,
        "valid": None,
        "errors": [],
        "status": "not_applicable",
    }
    if not bundle_root.is_dir():
        return payload

    handoff_path = bundle_root / "handoff_report.json"
    payload["present"] = bool(handoff_path.is_file())
    if not handoff_path.is_file():
        payload["status"] = "missing"
        return payload

    try:
        handoff_report = json.loads(handoff_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - validation boundary
        payload["valid"] = False
        payload["errors"] = [str(exc)]
        payload["status"] = "invalid"
        return payload

    if not isinstance(handoff_report, Mapping):
        payload["valid"] = False
        payload["errors"] = ["handoff_report.json must be a JSON object."]
        payload["status"] = "invalid"
        return payload

    errors = validate_deploy_bundle_handoff_report(handoff_report, bundle_dir=bundle_root)
    payload["valid"] = len(errors) == 0
    payload["errors"] = list(errors)
    payload["status"] = "valid" if payload["valid"] else "invalid"
    return payload


def _collect_bundle_blocking_reasons(
    *,
    bundle_root: Path,
    infer_config: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any],
    bundle_weights: Mapping[str, Any],
    handoff_report: Mapping[str, Any],
) -> tuple[list[str], str]:
    blocking_reasons: list[str] = []
    status = "partial"

    if not bundle_root.exists():
        return ["bundle_not_found"], "error"
    if not bundle_root.is_dir():
        return ["bundle_not_directory"], "error"

    if not bool(infer_config.get("present")):
        blocking_reasons.append("missing_infer_config")
    elif infer_config.get("valid") is not True:
        blocking_reasons.append("invalid_infer_config")

    if not bool(bundle_manifest.get("present")):
        blocking_reasons.append("missing_manifest")
    elif bundle_manifest.get("valid") is not True:
        blocking_reasons.append("invalid_manifest")
    else:
        manifest_payload = bundle_manifest.get("payload")
        if (
            isinstance(manifest_payload, Mapping)
            and manifest_payload.get("required_bundle_artifacts_present") is not True
        ):
            blocking_reasons.append("required_artifacts_missing")

    if bool(bundle_weights.get("applicable")) and bundle_weights.get("ready") is not True:
        blocking_reasons.append("bundle_weights_not_ready")
    if handoff_report.get("status") == "invalid":
        blocking_reasons.append("invalid_handoff_report")

    return blocking_reasons, status


def _bundle_contract_payload(bundle_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_payload = bundle_manifest.get("payload")
    runtime_policy = (
        normalize_deploy_bundle_runtime_policy(manifest_payload.get("runtime_policy", None))
        if isinstance(manifest_payload, Mapping)
        else normalize_deploy_bundle_runtime_policy(None)
    )
    contract = {
        "bundle_type": (
            str(manifest_payload.get("bundle_type"))
            if isinstance(manifest_payload, Mapping)
            and manifest_payload.get("bundle_type") is not None
            else None
        ),
        "input_contract": (
            dict(manifest_payload.get("input_contract", {}))
            if isinstance(manifest_payload, Mapping)
            and isinstance(manifest_payload.get("input_contract"), Mapping)
            else {}
        ),
        "output_contract": (
            dict(manifest_payload.get("output_contract", {}))
            if isinstance(manifest_payload, Mapping)
            and isinstance(manifest_payload.get("output_contract"), Mapping)
            else {}
        ),
        "runtime_policy": runtime_policy,
    }
    return contract


def evaluate_bundle(
    bundle_dir: str | Path,
    *,
    check_hashes: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    infer_config, bundle_manifest, bundle_weights, handoff_report = _evaluate_bundle_artifacts(
        bundle_root,
        check_hashes=bool(check_hashes),
    )
    blocking_reasons, status = _collect_bundle_blocking_reasons(
        bundle_root=bundle_root,
        infer_config=infer_config,
        bundle_manifest=bundle_manifest,
        bundle_weights=bundle_weights,
        handoff_report=handoff_report,
    )

    ready = len(blocking_reasons) == 0
    if ready:
        status = "ready"

    contract = _bundle_contract_payload(bundle_manifest)

    return {
        "schema_version": int(_BUNDLE_VALIDATE_SCHEMA_VERSION),
        "tool": "pyimgano-bundle",
        "command": "validate",
        "bundle_dir": str(bundle_root),
        "status": status,
        "ready": bool(ready),
        "exit_code": int(_validate_exit_code({"ready": bool(ready)})),
        "reason_codes": _reason_codes(blocking_reasons, mapping=_VALIDATE_REASON_CODE_MAP),
        "blocking_reasons": list(dict.fromkeys(str(item) for item in blocking_reasons)),
        "handoff_report_status": str(handoff_report.get("status", "not_applicable")),
        "next_action": (
            f"pyimgano bundle run {bundle_root} --image-dir /path/to/images --output-dir ./bundle_run --json"
            if ready
            else (
                f"pyimgano weights audit-bundle {bundle_root} --check-hashes --json"
                if "bundle_weights_not_ready" in blocking_reasons
                else (
                    f"pyimgano validate-infer-config {bundle_root / 'infer_config.json'}"
                    if "invalid_infer_config" in blocking_reasons
                    else f"pyimgano bundle validate {bundle_root} --json"
                )
            )
        ),
        "watch_command": (
            f"pyimgano bundle watch {bundle_root} --watch-dir /path/to/inbox --output-dir ./bundle_watch --once --json"
            if ready
            else None
        ),
        "contract": contract,
        "infer_config": infer_config,
        "bundle_manifest": bundle_manifest,
        "handoff_report": handoff_report,
        "bundle_weights": bundle_weights,
    }


def _default_input_record(
    *,
    resolved_path: str,
    image_path: str,
    source_root: Path | None = None,
    category: str | None = None,
    meta: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    path = Path(resolved_path)
    if record_id is None:
        if source_root is not None:
            try:
                record_id = path.resolve().relative_to(source_root.resolve()).as_posix()
            except Exception:
                record_id = path.name
        else:
            record_id = path.name

    return {
        "id": str(record_id),
        "image_path": str(image_path),
        "category": (str(category) if category is not None else None),
        "meta": (dict(meta) if meta is not None else None),
        "resolved_input_path": str(path),
    }


def _resolve_manifest_image_path(image_path: str, *, manifest_path: Path) -> str:
    path = Path(str(image_path))
    manifest_root = manifest_path.parent.resolve()
    if ".." in path.parts:
        raise ValueError(f"Manifest image_path contains path traversal: {image_path!r}")
    if path.is_absolute():
        candidate = path.resolve()
        try:
            candidate.relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError(
                f"Manifest image_path escapes manifest directory: {image_path!r}"
            ) from exc
        if not candidate.exists():
            raise FileNotFoundError(f"Manifest image_path not found: {candidate}")
        return str(candidate)

    candidate = (manifest_root / path).resolve()
    try:
        candidate.relative_to(manifest_root)
    except ValueError as exc:
        raise ValueError(f"Manifest image_path escapes manifest directory: {image_path!r}") from exc
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "Manifest image_path not found: "
        f"{image_path!r}. Tried {candidate} relative to manifest directory."
    )


def _records_from_paths(
    paths: Sequence[str | Path],
    *,
    source_root: Path | None,
    kind: str,
    empty_message: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        raise ValueError(empty_message)
    records = [
        _default_input_record(
            resolved_path=str(path),
            image_path=str(path),
            source_root=source_root,
        )
        for path in paths
    ]
    return records, {"kind": kind, "count": len(records)}


def _manifest_row_to_input_record(row: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    meta = row.get("meta")
    if meta is not None and not isinstance(meta, Mapping):
        raise ValueError("input_manifest.jsonl field 'meta' must be an object/dict.")
    raw_id = row.get("id")
    record_id = None
    if raw_id is not None and str(raw_id).strip():
        record_id = str(raw_id).strip()
    return _default_input_record(
        resolved_path=_resolve_manifest_image_path(
            str(row["image_path"]),
            manifest_path=manifest_path,
        ),
        image_path=str(row["image_path"]),
        source_root=None,
        category=(str(row["category"]) if row.get("category") is not None else None),
        meta=(dict(meta) if isinstance(meta, Mapping) else None),
        record_id=record_id,
    )


def _resolve_input_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if getattr(args, "image_dir", None) is not None:
        source_root = Path(str(args.image_dir))
        return _records_from_paths(
            collect_image_paths(str(source_root)),
            source_root=source_root,
            kind="image_dir",
            empty_message="No input images found in --image-dir.",
        )

    if getattr(args, "image", None) is not None:
        path = Path(str(args.image))
        records, _summary = _records_from_paths(
            collect_image_paths(str(path))[:1],
            source_root=None,
            kind="single_image",
            empty_message="No input images found in --image.",
        )
        return records, {"kind": "single_image", "count": 1}

    manifest_path = Path(str(args.input_manifest))
    records = [
        _manifest_row_to_input_record(row, manifest_path=manifest_path)
        for row in iter_manifest_rows(manifest_path)
    ]

    if not records:
        raise ValueError("No input records found in --input-manifest.")
    return records, {"kind": "input_manifest.jsonl", "count": len(records)}


def _rewrite_results_jsonl(
    *,
    output_dir: Path,
    results_path: Path,
    input_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_root = output_dir.resolve(strict=False)
    resolved_results_path = results_path.resolve(strict=False)
    try:
        resolved_results_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("results.jsonl path must stay within output_dir") from exc

    rows = [
        line
        for line in resolved_results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(input_records):
        raise RuntimeError(
            "results.jsonl record count does not match resolved input count. "
            f"results={len(rows)} inputs={len(input_records)}"
        )

    rewritten: list[dict[str, Any]] = []
    for line, input_record in zip(rows, input_records):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("results.jsonl must contain JSON object records.")
        record["id"] = str(input_record["id"])
        record["image_path"] = str(input_record["image_path"])
        record["category"] = input_record.get("category")
        record["meta"] = input_record.get("meta")
        rewritten.append(record)

    with resolved_results_path.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return rewritten


def _result_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    normal = 0
    anomalous = 0
    rejected = 0
    error = 0

    for record in records:
        if str(record.get("status", "")) == "error":
            error += 1
            continue
        label = record.get("label")
        if bool(record.get("rejected")):
            rejected += 1
        elif label == 0:
            normal += 1
        elif label == 1:
            anomalous += 1

    return {
        "normal": int(normal),
        "anomalous": int(anomalous),
        "rejected": int(rejected),
        "error": int(error),
    }


def _empty_result_summary() -> dict[str, int]:
    return {
        "normal": 0,
        "anomalous": 0,
        "rejected": 0,
        "error": 0,
    }


def _cli_batch_gate_thresholds(args: argparse.Namespace) -> dict[str, float | int | None]:
    return {
        "max_anomaly_rate": (
            float(args.max_anomaly_rate)
            if getattr(args, "max_anomaly_rate", None) is not None
            else None
        ),
        "max_reject_rate": (
            float(args.max_reject_rate)
            if getattr(args, "max_reject_rate", None) is not None
            else None
        ),
        "max_error_rate": (
            float(args.max_error_rate)
            if getattr(args, "max_error_rate", None) is not None
            else None
        ),
        "min_processed": (
            int(args.min_processed) if getattr(args, "min_processed", None) is not None else None
        ),
    }


def _manifest_batch_gate_thresholds(
    runtime_policy: Mapping[str, Any] | None,
) -> dict[str, float | int | None]:
    normalized = normalize_deploy_bundle_runtime_policy(runtime_policy)
    batch_gates = normalized.get("batch_gates", {})
    if not isinstance(batch_gates, Mapping):
        return {str(name): None for name in _BATCH_GATE_KEYS}
    return {str(name): batch_gates.get(name, None) for name in _BATCH_GATE_KEYS}


def _resolved_batch_gate_thresholds(
    args: argparse.Namespace,
    *,
    runtime_policy: Mapping[str, Any] | None,
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    cli_thresholds = _cli_batch_gate_thresholds(args)
    manifest_thresholds = _manifest_batch_gate_thresholds(runtime_policy)
    thresholds: dict[str, float | int | None] = {}
    sources: dict[str, str] = {}
    for name in _BATCH_GATE_KEYS:
        cli_value = cli_thresholds.get(name, None)
        manifest_value = manifest_thresholds.get(name, None)
        if cli_value is not None:
            thresholds[str(name)] = cli_value
            sources[str(name)] = "cli"
        elif manifest_value is not None:
            thresholds[str(name)] = manifest_value
            sources[str(name)] = "bundle_manifest"
        else:
            thresholds[str(name)] = None
            sources[str(name)] = "unset"
    return thresholds, sources


def _safe_rate(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _append_failed_gate(
    failed_gates: list[str],
    reason_codes: list[str],
    *,
    gate_name: str,
) -> None:
    failed_gates.append(gate_name)
    reason_codes.append(_RUN_BATCH_GATE_REASON_CODE_MAP[gate_name])


def _determine_batch_verdict(
    *, requested: bool, evaluated: bool, failed_gates: Sequence[str]
) -> str:
    if not requested:
        return "not_requested"
    if not evaluated:
        return "not_evaluated"
    if failed_gates:
        return "blocked"
    return "pass"


def _evaluate_batch_gates(
    *,
    args: argparse.Namespace,
    processed: int,
    result_summary: Mapping[str, Any] | None,
    evaluated: bool,
    runtime_policy: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, list[str]]:
    thresholds, sources = _resolved_batch_gate_thresholds(args, runtime_policy=runtime_policy)
    requested = any(value is not None for value in thresholds.values())

    counts = _empty_result_summary()
    if isinstance(result_summary, Mapping):
        counts = {
            "normal": int(result_summary.get("normal", 0)),
            "anomalous": int(result_summary.get("anomalous", 0)),
            "rejected": int(result_summary.get("rejected", 0)),
            "error": int(result_summary.get("error", 0)),
        }

    rates = {
        "anomaly_rate": _safe_rate(counts["anomalous"], processed),
        "reject_rate": _safe_rate(counts["rejected"], processed),
        "error_rate": _safe_rate(counts["error"], processed),
    }

    failed_gates: list[str] = []
    reason_codes: list[str] = []
    if evaluated and requested:
        min_processed = thresholds.get("min_processed")
        if min_processed is not None and int(processed) < int(min_processed):
            _append_failed_gate(failed_gates, reason_codes, gate_name="min_processed")

        max_anomaly_rate = thresholds.get("max_anomaly_rate")
        if max_anomaly_rate is not None and float(rates["anomaly_rate"]) > float(max_anomaly_rate):
            _append_failed_gate(failed_gates, reason_codes, gate_name="max_anomaly_rate")

        max_reject_rate = thresholds.get("max_reject_rate")
        if max_reject_rate is not None and float(rates["reject_rate"]) > float(max_reject_rate):
            _append_failed_gate(failed_gates, reason_codes, gate_name="max_reject_rate")

        max_error_rate = thresholds.get("max_error_rate")
        if max_error_rate is not None and float(rates["error_rate"]) > float(max_error_rate):
            _append_failed_gate(failed_gates, reason_codes, gate_name="max_error_rate")

    batch_verdict = _determine_batch_verdict(
        requested=bool(requested),
        evaluated=bool(evaluated),
        failed_gates=failed_gates,
    )

    return (
        _build_batch_gate_summary_helper(
            requested=bool(requested),
            evaluated=bool(evaluated),
            processed=int(processed),
            counts=counts,
            rates=rates,
            thresholds=thresholds,
            sources=sources,
            failed_gates=list(failed_gates),
        ),
        str(batch_verdict),
        list(reason_codes),
    )


def _bundle_supports_pixel_outputs(bundle_validation: Mapping[str, Any]) -> bool:
    contract = bundle_validation.get("contract", {})
    if not isinstance(contract, Mapping):
        return False
    output_contract = contract.get("output_contract", {})
    if not isinstance(output_contract, Mapping):
        return False
    return bool(output_contract.get("supports_pixel_outputs", False))


def _requested_pixel_exports(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "export_masks", False)
        or getattr(args, "export_overlays", False)
        or getattr(args, "export_defects_regions", False)
    )


def _resolved_optional_artifacts(
    args: argparse.Namespace, *, output_dir: Path
) -> dict[str, str | None]:
    return {
        "masks_dir": (
            str(output_dir / "masks") if bool(getattr(args, "export_masks", False)) else None
        ),
        "overlays_dir": (
            str(output_dir / "overlays") if bool(getattr(args, "export_overlays", False)) else None
        ),
        "defects_regions_jsonl": (
            str(output_dir / "defects_regions.jsonl")
            if bool(getattr(args, "export_defects_regions", False))
            else None
        ),
    }


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return FileHasher.compute_hash(str(path), algorithm="sha256")


def _tree_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_dir():
        return None

    parts: list[str] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel_path = item.relative_to(path).as_posix()
        sha256 = FileHasher.compute_hash(str(item), algorithm="sha256")
        parts.append(f"{rel_path}:{sha256}")

    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _artifact_digests(
    *,
    results_jsonl: Path | None,
    optional_artifacts: Mapping[str, str | None] | None,
) -> dict[str, str | None]:
    artifact_map = dict(optional_artifacts or {})
    defects_regions_jsonl = artifact_map.get("defects_regions_jsonl")
    masks_dir = artifact_map.get("masks_dir")
    overlays_dir = artifact_map.get("overlays_dir")
    return {
        "results_jsonl_sha256": _file_sha256(results_jsonl),
        "defects_regions_jsonl_sha256": (
            _file_sha256(Path(str(defects_regions_jsonl)))
            if defects_regions_jsonl is not None
            else None
        ),
        "masks_tree_sha256": (
            _tree_sha256(Path(str(masks_dir))) if masks_dir is not None else None
        ),
        "overlays_tree_sha256": (
            _tree_sha256(Path(str(overlays_dir))) if overlays_dir is not None else None
        ),
    }


def _build_run_report(
    *,
    bundle_validation: dict[str, Any],
    bundle_root: Path,
    output_dir: Path,
    input_summary: dict[str, Any],
    reason_codes: list[str],
    status: str,
    processed: int,
    results_jsonl: Path | None,
    result_records: list[dict[str, Any]] | None = None,
    optional_artifacts: Mapping[str, str | None] | None = None,
    batch_verdict: str = "not_requested",
    batch_gate_summary: Mapping[str, Any] | None = None,
    batch_gate_reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    records = list(result_records or [])
    report_path = output_dir / "run_report.json"
    optional_artifact_map = dict(optional_artifacts or {})
    exit_code = _run_exit_code(str(status))
    return {
        "schema_version": int(_BUNDLE_RUN_REPORT_SCHEMA_VERSION),
        "tool": "pyimgano-bundle",
        "command": "run",
        "bundle_dir": str(bundle_root),
        "output_dir": str(output_dir),
        "status": str(status),
        "ready": bool(status == "completed"),
        "exit_code": int(exit_code),
        "reason_codes": list(dict.fromkeys(str(item) for item in reason_codes)),
        "bundle_validation": {
            "status": bundle_validation.get("status"),
            "ready": bool(bundle_validation.get("ready", False)),
            "reason_codes": list(bundle_validation.get("reason_codes", [])),
        },
        "bundle_contract": dict(bundle_validation.get("contract", {})),
        "input_summary": _build_input_source_summary_helper(
            kind=str(input_summary.get("kind")),
            count=int(input_summary.get("count", 0)),
        ),
        "processed": int(processed),
        "batch_verdict": str(batch_verdict),
        "batch_gate_summary": dict(batch_gate_summary or {}),
        "batch_gate_reason_codes": list(
            dict.fromkeys(str(item) for item in (batch_gate_reason_codes or []))
        ),
        "result_summary": _result_summary(records),
        "artifact_digests": _artifact_digests(
            results_jsonl=results_jsonl,
            optional_artifacts=optional_artifacts,
        ),
        "artifacts": {
            "results_jsonl": (str(results_jsonl) if results_jsonl is not None else None),
            "run_report_json": str(report_path),
            "masks_dir": optional_artifact_map.get("masks_dir"),
            "overlays_dir": optional_artifact_map.get("overlays_dir"),
            "defects_regions_jsonl": optional_artifact_map.get("defects_regions_jsonl"),
        },
    }


def _write_run_report(report: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _emit_validate_payload(payload: dict[str, Any], *, json_output: bool) -> int:
    status_code = int(payload.get("exit_code", _validate_exit_code(payload)))
    if bool(json_output):
        return cli_output.emit_json(payload, status_code=status_code, indent=None)

    for line in bundle_rendering.format_bundle_validate_lines(payload):
        print(line)
    return status_code


def _emit_run_report(report: dict[str, Any], *, json_output: bool) -> int:
    status_code = int(report.get("exit_code", _run_exit_code(str(report.get("status")))))
    if bool(json_output):
        return cli_output.emit_json(report, status_code=status_code, indent=None)

    for line in bundle_rendering.format_bundle_run_lines(report):
        print(line)
    return status_code


def _emit_watch_report(report: dict[str, Any], *, json_output: bool) -> int:
    status_code = int(report.get("exit_code", 1))
    if bool(json_output):
        return cli_output.emit_json(report, status_code=status_code, indent=None)

    for line in bundle_rendering.format_bundle_watch_lines(report):
        print(line)
    return status_code


def _run_bundle(args: argparse.Namespace) -> dict[str, Any]:
    bundle_root = Path(str(args.bundle_dir))
    output_dir = Path(str(args.output_dir))
    bundle_validation = evaluate_bundle(bundle_root, check_hashes=bool(args.check_hashes))
    contract = bundle_validation.get("contract", {})
    runtime_policy = contract.get("runtime_policy", None) if isinstance(contract, Mapping) else None
    optional_artifacts = _resolved_optional_artifacts(args, output_dir=output_dir)
    batch_gate_summary, batch_verdict, batch_gate_reason_codes = _evaluate_batch_gates(
        args=args,
        processed=0,
        result_summary=None,
        evaluated=False,
        runtime_policy=runtime_policy,
    )

    if not bool(bundle_validation.get("ready")):
        report = _build_run_report(
            bundle_validation=bundle_validation,
            bundle_root=bundle_root,
            output_dir=output_dir,
            input_summary={"kind": "unresolved", "count": 0},
            reason_codes=list(bundle_validation.get("reason_codes", [])),
            status="blocked",
            processed=0,
            results_jsonl=None,
            result_records=[],
            optional_artifacts=optional_artifacts,
            batch_verdict=batch_verdict,
            batch_gate_summary=batch_gate_summary,
            batch_gate_reason_codes=batch_gate_reason_codes,
        )
        _write_run_report(report, output_dir=output_dir)
        return report

    try:
        input_records, input_summary = _resolve_input_records(args)
    except FileNotFoundError:
        reason_codes = [_RUN_REASON_CODE_MAP["input_not_found"]]
        report = _build_run_report(
            bundle_validation=bundle_validation,
            bundle_root=bundle_root,
            output_dir=output_dir,
            input_summary={"kind": "unresolved", "count": 0},
            reason_codes=reason_codes,
            status="failed",
            processed=0,
            results_jsonl=None,
            result_records=[],
            optional_artifacts=optional_artifacts,
            batch_verdict=batch_verdict,
            batch_gate_summary=batch_gate_summary,
            batch_gate_reason_codes=batch_gate_reason_codes,
        )
        _write_run_report(report, output_dir=output_dir)
        return report
    except Exception:
        reason_codes = [_RUN_REASON_CODE_MAP["invalid_input_manifest"]]
        report = _build_run_report(
            bundle_validation=bundle_validation,
            bundle_root=bundle_root,
            output_dir=output_dir,
            input_summary={"kind": "unresolved", "count": 0},
            reason_codes=reason_codes,
            status="failed",
            processed=0,
            results_jsonl=None,
            result_records=[],
            optional_artifacts=optional_artifacts,
            batch_verdict=batch_verdict,
            batch_gate_summary=batch_gate_summary,
            batch_gate_reason_codes=batch_gate_reason_codes,
        )
        _write_run_report(report, output_dir=output_dir)
        return report

    if _requested_pixel_exports(args) and not _bundle_supports_pixel_outputs(bundle_validation):
        report = _build_run_report(
            bundle_validation=bundle_validation,
            bundle_root=bundle_root,
            output_dir=output_dir,
            input_summary=input_summary,
            reason_codes=[_RUN_REASON_CODE_MAP["pixel_outputs_not_supported"]],
            status="blocked",
            processed=0,
            results_jsonl=None,
            result_records=[],
            optional_artifacts=optional_artifacts,
            batch_verdict=batch_verdict,
            batch_gate_summary=batch_gate_summary,
            batch_gate_reason_codes=batch_gate_reason_codes,
        )
        _write_run_report(report, output_dir=output_dir)
        return report

    results_path = output_dir / "results.jsonl"
    rc = run_bundle_inference_batch(
        BundleInferenceBatchRequest(
            bundle_dir=bundle_root,
            input_records=input_records,
            results_jsonl=results_path,
            defects_enabled=bool(_requested_pixel_exports(args)),
            masks_dir=(
                str(optional_artifacts["masks_dir"])
                if optional_artifacts.get("masks_dir") is not None
                else None
            ),
            overlays_dir=(
                str(optional_artifacts["overlays_dir"])
                if optional_artifacts.get("overlays_dir") is not None
                else None
            ),
            defects_regions_jsonl=(
                str(optional_artifacts["defects_regions_jsonl"])
                if optional_artifacts.get("defects_regions_jsonl") is not None
                else None
            ),
        )
    )
    if rc != 0:
        report = _build_run_report(
            bundle_validation=bundle_validation,
            bundle_root=bundle_root,
            output_dir=output_dir,
            input_summary=input_summary,
            reason_codes=[_RUN_REASON_CODE_MAP["inference_failed"]],
            status="failed",
            processed=0,
            results_jsonl=(results_path if results_path.exists() else None),
            result_records=[],
            optional_artifacts=optional_artifacts,
            batch_verdict=batch_verdict,
            batch_gate_summary=batch_gate_summary,
            batch_gate_reason_codes=batch_gate_reason_codes,
        )
        _write_run_report(report, output_dir=output_dir)
        return report

    result_records = _rewrite_results_jsonl(
        output_dir=output_dir,
        results_path=results_path,
        input_records=input_records,
    )
    result_summary = _result_summary(result_records)
    batch_gate_summary, batch_verdict, batch_gate_reason_codes = _evaluate_batch_gates(
        args=args,
        processed=len(result_records),
        result_summary=result_summary,
        evaluated=True,
        runtime_policy=runtime_policy,
    )
    status = "blocked" if batch_verdict == "blocked" else "completed"
    reason_codes = list(batch_gate_reason_codes if batch_verdict == "blocked" else [])
    report = _build_run_report(
        bundle_validation=bundle_validation,
        bundle_root=bundle_root,
        output_dir=output_dir,
        input_summary=input_summary,
        reason_codes=reason_codes,
        status=status,
        processed=len(result_records),
        results_jsonl=results_path,
        result_records=result_records,
        optional_artifacts=optional_artifacts,
        batch_verdict=batch_verdict,
        batch_gate_summary=batch_gate_summary,
        batch_gate_reason_codes=batch_gate_reason_codes,
    )
    _write_run_report(report, output_dir=output_dir)
    return report


def _watch_request_from_args(args: argparse.Namespace) -> bundle_watch_service.BundleWatchRequest:
    return bundle_watch_service.BundleWatchRequest(
        bundle_dir=str(args.bundle_dir),
        watch_dir=str(args.watch_dir),
        output_dir=str(args.output_dir),
        poll_seconds=float(args.poll_seconds),
        settle_seconds=float(args.settle_seconds),
        once=bool(args.once),
        state_file=(str(args.state_file) if getattr(args, "state_file", None) is not None else None),
        check_hashes=bool(args.check_hashes),
        export_masks=bool(args.export_masks),
        export_overlays=bool(args.export_overlays),
        export_defects_regions=bool(args.export_defects_regions),
        webhook_url=(
            str(args.webhook_url) if getattr(args, "webhook_url", None) is not None else None
        ),
        webhook_bearer_token=_resolve_secret_from_cli_or_env(
            value=(
                str(args.webhook_bearer_token)
                if getattr(args, "webhook_bearer_token", None) is not None
                else None
            ),
            env_var=(
                str(args.webhook_bearer_token_env)
                if getattr(args, "webhook_bearer_token_env", None) is not None
                else None
            ),
            option_name="--webhook-bearer-token",
            env_option_name="--webhook-bearer-token-env",
        ),
        webhook_signing_secret=_resolve_secret_from_cli_or_env(
            value=(
                str(args.webhook_signing_secret)
                if getattr(args, "webhook_signing_secret", None) is not None
                else None
            ),
            env_var=(
                str(args.webhook_signing_secret_env)
                if getattr(args, "webhook_signing_secret_env", None) is not None
                else None
            ),
            option_name="--webhook-signing-secret",
            env_option_name="--webhook-signing-secret-env",
        ),
        webhook_headers={
            str(key): str(value)
            for key, value in (list(getattr(args, "webhook_header", []) or []))
        },
        webhook_timeout_seconds=float(args.webhook_timeout_seconds),
        max_anomaly_rate=(
            float(args.max_anomaly_rate) if getattr(args, "max_anomaly_rate", None) is not None else None
        ),
        max_reject_rate=(
            float(args.max_reject_rate) if getattr(args, "max_reject_rate", None) is not None else None
        ),
        max_error_rate=(
            float(args.max_error_rate) if getattr(args, "max_error_rate", None) is not None else None
        ),
        min_processed=(
            int(args.min_processed) if getattr(args, "min_processed", None) is not None else None
        ),
    )


def _watch_bundle(args: argparse.Namespace) -> dict[str, Any]:
    request = _watch_request_from_args(args)
    report = bundle_watch_service.run_bundle_watch_once(request)
    if bool(request.once):
        return report

    while True:
        try:
            time.sleep(float(request.poll_seconds))
            report = bundle_watch_service.run_bundle_watch_once(request)
        except KeyboardInterrupt:
            break
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if str(args.command) == "validate":
        payload = evaluate_bundle(
            str(args.bundle_dir),
            check_hashes=bool(getattr(args, "check_hashes", False)),
        )
        return _emit_validate_payload(payload, json_output=bool(getattr(args, "json", False)))

    if str(args.command) == "watch":
        try:
            report = _watch_bundle(args)
        except ValueError as exc:
            parser.error(str(exc))
        return _emit_watch_report(report, json_output=bool(getattr(args, "json", False)))

    report = _run_bundle(args)
    return _emit_run_report(report, json_output=bool(getattr(args, "json", False)))


__all__ = ["evaluate_bundle", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
