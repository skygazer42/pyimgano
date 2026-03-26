from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyimgano.cli_output as cli_output
import pyimgano.infer_cli as infer_cli
from pyimgano.datasets.manifest_tools import iter_manifest_rows
from pyimgano.inference.validate_infer_config import validate_infer_config_file
from pyimgano.reporting.deploy_bundle import validate_deploy_bundle_manifest
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
    out: list[str] = []
    for reason in blocking_reasons:
        code = mapping.get(str(reason))
        if code is not None and code not in out:
            out.append(code)
    return out


def _validate_exit_code(payload: Mapping[str, Any]) -> int:
    return 0 if bool(payload.get("ready")) else 1


def _run_exit_code(status: str) -> int:
    return 0 if str(status) == "completed" else 1


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not bundle_root.is_dir():
        return {}, {}, _default_bundle_weights_payload(bundle_root)
    infer_config = _evaluate_infer_config(bundle_root)
    bundle_manifest = _evaluate_bundle_manifest(bundle_root, check_hashes=bool(check_hashes))
    bundle_weights = _bundle_weights_payload(bundle_root, check_hashes=bool(check_hashes))
    return infer_config, bundle_manifest, bundle_weights


def _collect_bundle_blocking_reasons(
    *,
    bundle_root: Path,
    infer_config: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any],
    bundle_weights: Mapping[str, Any],
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

    return blocking_reasons, status


def _bundle_contract_payload(bundle_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_payload = bundle_manifest.get("payload")
    return {
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
    }


def evaluate_bundle(
    bundle_dir: str | Path,
    *,
    check_hashes: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    infer_config, bundle_manifest, bundle_weights = _evaluate_bundle_artifacts(
        bundle_root,
        check_hashes=bool(check_hashes),
    )
    blocking_reasons, status = _collect_bundle_blocking_reasons(
        bundle_root=bundle_root,
        infer_config=infer_config,
        bundle_manifest=bundle_manifest,
        bundle_weights=bundle_weights,
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
        "contract": contract,
        "infer_config": infer_config,
        "bundle_manifest": bundle_manifest,
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
            infer_cli._collect_image_paths(str(source_root)),
            source_root=source_root,
            kind="image_dir",
            empty_message="No input images found in --image-dir.",
        )

    if getattr(args, "image", None) is not None:
        path = Path(str(args.image))
        records, _summary = _records_from_paths(
            infer_cli._collect_image_paths(str(path))[:1],
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


def _batch_gate_thresholds(args: argparse.Namespace) -> dict[str, float | int | None]:
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
) -> tuple[dict[str, Any], str, list[str]]:
    thresholds = _batch_gate_thresholds(args)
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
        {
            "requested": bool(requested),
            "evaluated": bool(evaluated),
            "processed": int(processed),
            "counts": counts,
            "rates": rates,
            "thresholds": thresholds,
            "failed_gates": list(failed_gates),
        },
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
        "input_summary": {
            "kind": str(input_summary.get("kind")),
            "count": int(input_summary.get("count", 0)),
        },
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

    print(f"bundle_dir={payload.get('bundle_dir')}")
    print(f"status={payload.get('status')}")
    print(f"ready={str(bool(payload.get('ready'))).lower()}")
    for code in payload.get("reason_codes", []):
        print(f"reason_code={code}")
    contract = payload.get("contract", {})
    if isinstance(contract, Mapping):
        bundle_type = contract.get("bundle_type")
        if bundle_type is not None:
            print(f"bundle_type={bundle_type}")
    return status_code


def _emit_run_report(report: dict[str, Any], *, json_output: bool) -> int:
    status_code = int(report.get("exit_code", _run_exit_code(str(report.get("status")))))
    if bool(json_output):
        return cli_output.emit_json(report, status_code=status_code, indent=None)

    print(f"bundle_dir={report.get('bundle_dir')}")
    print(f"output_dir={report.get('output_dir')}")
    print(f"status={report.get('status')}")
    print(f"processed={report.get('processed')}")
    if report.get("batch_verdict") is not None:
        print(f"batch_verdict={report.get('batch_verdict')}")
    for code in report.get("reason_codes", []):
        print(f"reason_code={code}")
    artifacts = report.get("artifacts", {})
    if isinstance(artifacts, Mapping) and artifacts.get("results_jsonl") is not None:
        print(f"results_jsonl={artifacts.get('results_jsonl')}")
    return status_code


def _run_bundle(args: argparse.Namespace) -> dict[str, Any]:
    bundle_root = Path(str(args.bundle_dir))
    output_dir = Path(str(args.output_dir))
    bundle_validation = evaluate_bundle(bundle_root, check_hashes=bool(args.check_hashes))
    optional_artifacts = _resolved_optional_artifacts(args, output_dir=output_dir)
    batch_gate_summary, batch_verdict, batch_gate_reason_codes = _evaluate_batch_gates(
        args=args,
        processed=0,
        result_summary=None,
        evaluated=False,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    infer_argv = [
        "--infer-config",
        str(bundle_root / "infer_config.json"),
        "--save-jsonl",
        str(results_path),
    ]
    for input_record in input_records:
        infer_argv.extend(["--input", str(input_record["resolved_input_path"])])

    if _requested_pixel_exports(args):
        infer_argv.append("--defects")
    if optional_artifacts.get("masks_dir") is not None:
        infer_argv.extend(["--save-masks", str(optional_artifacts["masks_dir"])])
    if optional_artifacts.get("overlays_dir") is not None:
        infer_argv.extend(["--save-overlays", str(optional_artifacts["overlays_dir"])])
    if optional_artifacts.get("defects_regions_jsonl") is not None:
        infer_argv.extend(
            ["--defects-regions-jsonl", str(optional_artifacts["defects_regions_jsonl"])]
        )

    rc = int(infer_cli.main(infer_argv))
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if str(args.command) == "validate":
        payload = evaluate_bundle(
            str(args.bundle_dir),
            check_hashes=bool(getattr(args, "check_hashes", False)),
        )
        return _emit_validate_payload(payload, json_output=bool(getattr(args, "json", False)))

    report = _run_bundle(args)
    return _emit_run_report(report, json_output=bool(getattr(args, "json", False)))


__all__ = ["evaluate_bundle", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
