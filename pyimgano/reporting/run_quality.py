from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyimgano.reporting.calibration_card import validate_calibration_card_payload
from pyimgano.reporting.deploy_bundle import validate_deploy_bundle_manifest
from pyimgano.weights.bundle_audit import evaluate_bundle_weights_audit


@dataclass(frozen=True)
class _ArtifactSpec:
    name: str
    rel_path: str
    required: bool


_ARTIFACT_SPECS = (
    _ArtifactSpec(name="report", rel_path="report.json", required=True),
    _ArtifactSpec(name="config", rel_path="config.json", required=True),
    _ArtifactSpec(name="environment", rel_path="environment.json", required=True),
    _ArtifactSpec(name="infer_config", rel_path="artifacts/infer_config.json", required=False),
    _ArtifactSpec(
        name="operator_contract",
        rel_path="artifacts/operator_contract.json",
        required=False,
    ),
    _ArtifactSpec(
        name="calibration_card",
        rel_path="artifacts/calibration_card.json",
        required=False,
    ),
    _ArtifactSpec(
        name="deploy_bundle_manifest",
        rel_path="deploy_bundle/bundle_manifest.json",
        required=False,
    ),
)


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def _report_dataset_readiness(root: Path) -> dict[str, Any] | None:
    report_path = root / _ARTIFACT_SPECS[0].rel_path
    if not report_path.exists():
        return None
    try:
        report = _load_json_dict(report_path)
    except Exception:
        return None
    readiness = report.get("dataset_readiness", None)
    if not isinstance(readiness, Mapping):
        return None
    payload = dict(readiness)
    issue_codes = payload.get("issue_codes", None)
    if not isinstance(issue_codes, list):
        payload["issue_codes"] = []
    issue_details = payload.get("issue_details", None)
    if not isinstance(issue_details, list):
        payload["issue_details"] = []
    return payload


def _validation_payload(rel_path: str) -> dict[str, Any]:
    return {
        "path": rel_path,
        "present": False,
        "valid": None,
        "errors": [],
        "warnings": [],
    }


def _extract_threshold_value(payload: Mapping[str, Any]) -> float | None:
    postprocess = payload.get("postprocess", None)
    if isinstance(postprocess, Mapping):
        image_threshold = postprocess.get("image_threshold", None)
        if isinstance(image_threshold, Mapping):
            nested = image_threshold.get("threshold", None)
            if isinstance(nested, (int, float)):
                return float(nested)

    threshold = payload.get("threshold", None)
    if isinstance(threshold, (int, float)):
        return float(threshold)

    image_threshold = payload.get("image_threshold", None)
    if isinstance(image_threshold, Mapping):
        nested = image_threshold.get("threshold", None)
        if isinstance(nested, (int, float)):
            return float(nested)

    return None


def _infer_declares_prediction_policy(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("prediction", None), Mapping):
        return True

    postprocess = payload.get("postprocess", None)
    if not isinstance(postprocess, Mapping):
        return False

    return isinstance(postprocess.get("review_policy", None), Mapping)


def _extract_split_fingerprint_sha256(payload: Mapping[str, Any]) -> str | None:
    split_fingerprint = payload.get("split_fingerprint", None)
    if not isinstance(split_fingerprint, Mapping):
        return None

    sha256 = split_fingerprint.get("sha256", None)
    if not isinstance(sha256, str) or not sha256.strip():
        return None
    return str(sha256)


def _evaluate_calibration_audit(
    *,
    root: Path,
    infer_config_present: bool,
    calibration_present: bool,
    calibration_valid: bool | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "present": bool(calibration_present),
        "valid": calibration_valid,
        "matching_threshold": None,
        "matching_split_fingerprint": None,
        "has_threshold_context": False,
        "has_split_fingerprint": False,
        "has_prediction_policy": False,
        "warnings": [],
    }
    warnings: list[str] = []

    infer_payload = _load_optional_audit_payload(
        root / "artifacts" / "infer_config.json",
        label="infer_config.json",
        enabled=bool(infer_config_present),
        warnings=warnings,
    )
    calibration_payload = _load_optional_audit_payload(
        root / "artifacts" / "calibration_card.json",
        label="calibration_card.json",
        enabled=bool(calibration_present and calibration_valid is not False),
        warnings=warnings,
    )

    _populate_calibration_payload_flags(payload, calibration_payload)
    _append_calibration_comparison_warnings(
        payload,
        warnings,
        calibration_valid=calibration_valid,
        infer_payload=infer_payload,
        calibration_payload=calibration_payload,
    )

    payload["warnings"] = warnings
    return payload


def _evaluate_bundle_weights_audit(
    bundle_dir: str | Path,
    *,
    check_hashes: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    if not bundle_root.exists() or not bundle_root.is_dir():
        return {
            "bundle_dir": str(bundle_root),
            "present": False,
            "valid": None,
            "cross_checked": False,
            "missing_required": [],
            "warnings": [],
            "errors": [],
            "status": "partial",
            "ready": False,
            "model_card": _validation_payload("model_card.json"),
            "weights_manifest": _validation_payload("weights_manifest.json"),
            "trust_summary": {
                "status": "partial",
                "trust_signals": {
                    "file_refs_checked": True,
                    "hashes_checked": bool(check_hashes),
                    "has_model_card": False,
                    "model_card_valid": False,
                    "has_weights_manifest": False,
                    "weights_manifest_valid": False,
                    "has_cross_checked_manifest": False,
                    "has_weights_sha256": False,
                    "has_weights_source": False,
                    "has_weights_license": False,
                    "has_deployment_runtime": False,
                    "has_manifest_link": False,
                    "manifest_has_entries": False,
                    "manifest_all_entries_have_sha256": False,
                    "manifest_all_entries_have_source": False,
                    "manifest_all_entries_have_license": False,
                    "manifest_all_entries_have_runtime": False,
                },
                "degraded_by": ["missing_model_card", "missing_weights_manifest"],
                "audit_refs": {},
            },
        }
    return evaluate_bundle_weights_audit(bundle_dir, check_hashes=bool(check_hashes))


def _evaluate_operator_contract_audit(
    *,
    root: Path,
    infer_config_present: bool,
    operator_contract_present: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "present": bool(operator_contract_present),
        "has_infer_contract": False,
        "matching_payload": None,
        "warnings": [],
    }
    warnings: list[str] = []

    if not bool(operator_contract_present):
        return payload

    operator_contract_payload: dict[str, Any] | None = None
    operator_contract_path = root / "artifacts" / "operator_contract.json"
    try:
        operator_contract_payload = _load_json_dict(operator_contract_path)
    except Exception as exc:  # noqa: BLE001 - reporting boundary
        warnings.append(f"Failed to inspect operator_contract.json for audit: {exc}")
        payload["matching_payload"] = False
        payload["warnings"] = warnings
        return payload

    if not bool(infer_config_present):
        warnings.append(
            "Operator contract audit warning: infer_config.json is missing while operator_contract.json exists."
        )
        payload["matching_payload"] = False
        payload["warnings"] = warnings
        return payload

    infer_payload: dict[str, Any] | None = None
    infer_path = root / "artifacts" / "infer_config.json"
    try:
        infer_payload = _load_json_dict(infer_path)
    except Exception as exc:  # noqa: BLE001 - reporting boundary
        warnings.append(f"Failed to inspect infer_config.json for operator contract audit: {exc}")
        payload["matching_payload"] = False
        payload["warnings"] = warnings
        return payload

    infer_contract = infer_payload.get("operator_contract", None) if infer_payload else None
    if not isinstance(infer_contract, Mapping):
        warnings.append(
            "Operator contract audit warning: infer_config.json is missing operator_contract metadata."
        )
        payload["matching_payload"] = False
        payload["warnings"] = warnings
        return payload

    payload["has_infer_contract"] = True
    matches = dict(infer_contract) == dict(operator_contract_payload)
    payload["matching_payload"] = bool(matches)
    if not bool(matches):
        warnings.append(
            "Operator contract audit warning: operator_contract mismatch between infer_config.json and operator_contract.json."
        )
    payload["warnings"] = warnings
    return payload


def _build_audit_refs(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    weights_audit: Mapping[str, Any],
) -> dict[str, str]:
    refs: dict[str, str] = {}
    key_map = {
        "report": "report_json",
        "config": "config_json",
        "environment": "environment_json",
        "infer_config": "infer_config_json",
        "operator_contract": "operator_contract_json",
        "calibration_card": "calibration_card_json",
        "deploy_bundle_manifest": "deploy_bundle_manifest_json",
    }
    for name, ref_key in key_map.items():
        payload = artifacts.get(name, {})
        if bool(payload.get("present")) and isinstance(payload.get("path"), str):
            refs[str(ref_key)] = str(payload["path"])

    model_card = weights_audit.get("model_card", {})
    if bool(model_card.get("present")):
        refs["bundle_model_card_json"] = "deploy_bundle/model_card.json"

    weights_manifest = weights_audit.get("weights_manifest", {})
    if bool(weights_manifest.get("present")):
        refs["bundle_weights_manifest_json"] = "deploy_bundle/weights_manifest.json"

    return refs


def _build_trust_summary(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    missing_required: list[str],
    core_complete: bool,
    audited_complete: bool,
    deployable_complete: bool,
    warnings: list[str],
    calibration_audit: Mapping[str, Any],
    operator_contract_audit: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any],
    weights_audit: Mapping[str, Any],
    has_bundle_operator_contract: bool,
    has_bundle_operator_contract_consistent: bool,
    has_bundle_operator_contract_digests_valid: bool,
    has_bundle_operator_contract_error: bool,
    has_bundle_operator_contract_digest_error: bool,
) -> dict[str, Any]:
    trust_signals = {
        "has_core_artifacts": bool(core_complete),
        "has_infer_config": bool(artifacts.get("infer_config", {}).get("present")),
        "has_operator_contract": bool(artifacts.get("operator_contract", {}).get("present")),
        "has_operator_contract_consistent": bool(
            operator_contract_audit.get("matching_payload") is True
        ),
        "has_bundle_operator_contract": bool(has_bundle_operator_contract),
        "has_bundle_operator_contract_consistent": bool(has_bundle_operator_contract_consistent),
        "has_bundle_operator_contract_digests_valid": bool(
            has_bundle_operator_contract_digests_valid
        ),
        "has_calibration_card": bool(artifacts.get("calibration_card", {}).get("present")),
        "has_threshold_context": bool(calibration_audit.get("has_threshold_context")),
        "has_split_fingerprint": bool(calibration_audit.get("has_split_fingerprint")),
        "has_prediction_policy": bool(calibration_audit.get("has_prediction_policy")),
        "has_deploy_bundle_manifest": bool(bundle_manifest.get("present")),
        "has_valid_deploy_bundle_manifest": bool(bundle_manifest.get("valid") is True),
        "has_bundle_weights_audit": bool(weights_audit.get("present")),
        "has_valid_bundle_weights_audit": bool(weights_audit.get("valid") is True),
    }
    status_reasons = _trust_status_reasons(
        artifacts=artifacts,
        core_complete=core_complete,
        audited_complete=audited_complete,
        deployable_complete=deployable_complete,
        warnings=warnings,
        operator_contract_audit=operator_contract_audit,
        weights_audit=weights_audit,
    )
    degraded_by = _trust_degraded_by(
        missing_required=missing_required,
        warnings=warnings,
        bundle_manifest=bundle_manifest,
        weights_audit=weights_audit,
        operator_contract_audit=operator_contract_audit,
        has_bundle_operator_contract_error=has_bundle_operator_contract_error,
        has_bundle_operator_contract_digest_error=has_bundle_operator_contract_digest_error,
    )
    status = _trust_status(
        artifacts=artifacts,
        core_complete=core_complete,
        audited_complete=audited_complete,
        degraded_by=degraded_by,
    )

    return {
        "status": status,
        "trust_signals": trust_signals,
        "status_reasons": list(dict.fromkeys(str(item) for item in status_reasons)),
        "degraded_by": list(dict.fromkeys(str(item) for item in degraded_by)),
        "audit_refs": _build_audit_refs(artifacts=artifacts, weights_audit=weights_audit),
    }


def _load_optional_audit_payload(
    path: Path,
    *,
    label: str,
    enabled: bool,
    warnings: list[str],
) -> dict[str, Any] | None:
    if not enabled:
        return None
    try:
        return _load_json_dict(path)
    except Exception as exc:  # noqa: BLE001 - reporting boundary
        warnings.append(f"Failed to inspect {label} for calibration audit: {exc}")
        return None


def _populate_calibration_payload_flags(
    payload: dict[str, Any],
    calibration_payload: Mapping[str, Any] | None,
) -> None:
    if not isinstance(calibration_payload, Mapping):
        return
    payload["has_threshold_context"] = isinstance(
        calibration_payload.get("threshold_context", None), Mapping
    )
    payload["has_split_fingerprint"] = (
        _extract_split_fingerprint_sha256(calibration_payload) is not None
    )
    payload["has_prediction_policy"] = isinstance(
        calibration_payload.get("prediction_policy", None),
        Mapping,
    )


def _append_calibration_comparison_warnings(
    payload: dict[str, Any],
    warnings: list[str],
    *,
    calibration_valid: bool | None,
    infer_payload: Mapping[str, Any] | None,
    calibration_payload: Mapping[str, Any] | None,
) -> None:
    if (
        calibration_valid is not True
        or not isinstance(infer_payload, Mapping)
        or not isinstance(calibration_payload, Mapping)
    ):
        return

    infer_threshold = _extract_threshold_value(infer_payload)
    calibration_threshold = _extract_threshold_value(calibration_payload)
    if infer_threshold is not None and calibration_threshold is not None:
        matches = abs(float(infer_threshold) - float(calibration_threshold)) <= 1e-12
        payload["matching_threshold"] = bool(matches)
        if not matches:
            warnings.append(
                "Calibration audit warning: threshold mismatch between infer_config.json and calibration_card.json."
            )

    infer_split_sha256 = _extract_split_fingerprint_sha256(infer_payload)
    calibration_split_sha256 = _extract_split_fingerprint_sha256(calibration_payload)
    if infer_split_sha256 is not None and calibration_split_sha256 is not None:
        matches = infer_split_sha256 == calibration_split_sha256
        payload["matching_split_fingerprint"] = bool(matches)
        if not matches:
            warnings.append(
                "Calibration audit warning: split_fingerprint mismatch between infer_config.json and calibration_card.json."
            )
    elif infer_split_sha256 is not None:
        warnings.append(
            "Calibration audit warning: calibration_card.json is missing split_fingerprint metadata."
        )

    if _infer_declares_prediction_policy(infer_payload) and not bool(
        payload["has_prediction_policy"]
    ):
        warnings.append(
            "Calibration audit warning: calibration_card.json is missing prediction_policy metadata."
        )
    if not bool(payload["has_threshold_context"]):
        warnings.append(
            "Calibration audit warning: calibration_card.json is missing threshold_context metadata."
        )


def _trust_status_reasons(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    core_complete: bool,
    audited_complete: bool,
    deployable_complete: bool,
    warnings: Sequence[str],
    operator_contract_audit: Mapping[str, Any],
    weights_audit: Mapping[str, Any],
) -> list[str]:
    status_reasons: list[str] = []
    if bool(core_complete):
        status_reasons.append("core_artifacts_present")
    if bool(audited_complete):
        status_reasons.append("calibration_audit_consistent")
    elif bool(artifacts.get("infer_config", {}).get("present")) and bool(
        artifacts.get("calibration_card", {}).get("present")
    ):
        status_reasons.append("calibration_audit_incomplete")
    if bool(artifacts.get("operator_contract", {}).get("present")):
        if operator_contract_audit.get("matching_payload") is True:
            status_reasons.append("operator_contract_consistent")
        else:
            status_reasons.append("operator_contract_incomplete")
    if bool(deployable_complete):
        status_reasons.append("deploy_bundle_audited")
    elif bool(artifacts.get("deploy_bundle_manifest", {}).get("present")) or bool(
        weights_audit.get("present")
    ):
        status_reasons.append("deploy_bundle_incomplete")
    if warnings:
        status_reasons.append("warnings_present")
    return status_reasons


def _append_unique_reason(degraded_by: list[str], reason: str) -> None:
    if reason not in degraded_by:
        degraded_by.append(reason)


def _warning_degradation_reasons(warnings: Sequence[str]) -> list[str]:
    degraded_by: list[str] = []
    for warning in warnings:
        text = str(warning).lower()
        if "threshold_context" in text:
            _append_unique_reason(degraded_by, "missing_threshold_context")
        if "split_fingerprint" in text:
            _append_unique_reason(degraded_by, "missing_split_fingerprint")
        if "prediction_policy" in text:
            _append_unique_reason(degraded_by, "missing_prediction_policy")
        if "threshold mismatch" in text:
            _append_unique_reason(degraded_by, "threshold_mismatch")
        if "operator contract" in text:
            _append_unique_reason(degraded_by, "operator_contract_mismatch")
    return degraded_by


def _trust_degraded_by(
    *,
    missing_required: Sequence[str],
    warnings: Sequence[str],
    bundle_manifest: Mapping[str, Any],
    weights_audit: Mapping[str, Any],
    operator_contract_audit: Mapping[str, Any],
    has_bundle_operator_contract_error: bool,
    has_bundle_operator_contract_digest_error: bool,
) -> list[str]:
    degraded_by = _warning_degradation_reasons(warnings)
    if missing_required:
        degraded_by.append("missing_required_artifacts")

    if bundle_manifest.get("present") and bundle_manifest.get("valid") is False:
        degraded_by.append("invalid_bundle_manifest")
    if weights_audit.get("present") and weights_audit.get("valid") is False:
        degraded_by.append("invalid_bundle_weights_audit")
    if (
        bool(operator_contract_audit.get("present"))
        and operator_contract_audit.get("matching_payload") is False
        and "operator_contract_mismatch" not in degraded_by
    ):
        degraded_by.append("operator_contract_mismatch")
    if bool(has_bundle_operator_contract_error):
        degraded_by.append("operator_contract_bundle_mismatch")
    if bool(has_bundle_operator_contract_digest_error):
        degraded_by.append("operator_contract_bundle_digest_mismatch")
    return degraded_by


def _trust_status(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    core_complete: bool,
    audited_complete: bool,
    degraded_by: Sequence[str],
) -> str:
    report_present = bool(artifacts.get("report", {}).get("present"))
    if not report_present:
        return "broken"
    if not bool(core_complete):
        return "partial"
    if bool(audited_complete) and not degraded_by:
        return "trust-signaled"
    return "partial"


def _artifact_inventory(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for spec in _ARTIFACT_SPECS:
        path = root / spec.rel_path
        present = path.is_file()
        artifact_payload: dict[str, Any] = {
            "path": spec.rel_path,
            "required": bool(spec.required),
            "present": bool(present),
        }
        if spec.name == "calibration_card":
            artifact_payload["valid"] = None
            artifact_payload["errors"] = []
        artifacts[spec.name] = artifact_payload
        if present:
            continue
        if spec.required:
            missing_required.append(spec.rel_path)
        else:
            missing_optional.append(spec.rel_path)
    return artifacts, missing_required, missing_optional


def _calibration_artifact_status(
    root: Path,
    *,
    artifacts: Mapping[str, dict[str, Any]],
) -> bool | None:
    if not bool(artifacts["calibration_card"]["present"]):
        return None

    calibration_path = root / "artifacts" / "calibration_card.json"
    try:
        calibration_payload = _load_json_dict(calibration_path)
    except Exception as exc:  # noqa: BLE001 - reporting boundary
        artifacts["calibration_card"]["valid"] = False
        artifacts["calibration_card"]["errors"] = [str(exc)]
        return False

    errors = validate_calibration_card_payload(calibration_payload)
    artifacts["calibration_card"]["valid"] = len(errors) == 0
    artifacts["calibration_card"]["errors"] = list(errors)
    return len(errors) == 0


def _bundle_manifest_status(
    root: Path,
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    check_bundle_hashes: bool,
) -> dict[str, Any]:
    bundle_manifest_payload = {
        "present": bool(artifacts["deploy_bundle_manifest"]["present"]),
        "valid": None,
        "errors": [],
    }
    if not bool(artifacts["deploy_bundle_manifest"]["present"]):
        return bundle_manifest_payload

    manifest_path = root / "deploy_bundle" / "bundle_manifest.json"
    try:
        manifest = _load_json_dict(manifest_path)
    except Exception as exc:  # noqa: BLE001 - reporting boundary
        bundle_manifest_payload["valid"] = False
        bundle_manifest_payload["errors"] = [str(exc)]
        return bundle_manifest_payload

    errors = validate_deploy_bundle_manifest(
        manifest,
        bundle_dir=root / "deploy_bundle",
        check_hashes=bool(check_bundle_hashes),
    )
    bundle_manifest_payload["valid"] = len(errors) == 0
    bundle_manifest_payload["errors"] = list(errors)
    return bundle_manifest_payload


def _bundle_operator_contract_flags(
    root: Path,
    *,
    bundle_manifest_payload: Mapping[str, Any],
) -> tuple[bool, bool, bool, bool, bool]:
    has_bundle_operator_contract = bool(
        (root / "deploy_bundle" / "operator_contract.json").is_file()
    )
    bundle_operator_contract_digest_error_present = any(
        "operator_contract_digests" in str(item).lower()
        for item in bundle_manifest_payload["errors"]
    )
    bundle_operator_contract_error_present = any(
        "operator_contract" in str(item).lower() or "operator contract" in str(item).lower()
        for item in bundle_manifest_payload["errors"]
    )
    has_bundle_operator_contract_consistent = (
        bool(has_bundle_operator_contract)
        and bool(bundle_manifest_payload.get("present"))
        and not bool(bundle_operator_contract_error_present)
    )
    has_bundle_operator_contract_digests_valid = (
        bool(has_bundle_operator_contract)
        and bool(bundle_manifest_payload.get("present"))
        and not bool(bundle_operator_contract_digest_error_present)
    )
    return (
        has_bundle_operator_contract,
        has_bundle_operator_contract_consistent,
        has_bundle_operator_contract_digests_valid,
        bundle_operator_contract_error_present,
        bundle_operator_contract_digest_error_present,
    )


def _quality_status_and_score(
    *,
    report_present: bool,
    core_complete: bool,
    audited_complete: bool,
    deployable_complete: bool,
) -> tuple[str, float]:
    if deployable_complete:
        return "deployable", 1.0
    if audited_complete:
        return "audited", 0.75
    if core_complete:
        return "reproducible", 0.5
    if report_present:
        return "partial", 0.25
    return "broken", 0.0


def evaluate_run_quality(
    run_dir: str | Path,
    *,
    check_bundle_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)

    artifacts, missing_required, missing_optional = _artifact_inventory(root)
    calibration_valid = _calibration_artifact_status(root, artifacts=artifacts)

    calibration_audit = _evaluate_calibration_audit(
        root=root,
        infer_config_present=bool(artifacts["infer_config"]["present"]),
        calibration_present=bool(artifacts["calibration_card"]["present"]),
        calibration_valid=calibration_valid,
    )
    operator_contract_audit = _evaluate_operator_contract_audit(
        root=root,
        infer_config_present=bool(artifacts["infer_config"]["present"]),
        operator_contract_present=bool(artifacts["operator_contract"]["present"]),
    )

    bundle_manifest_payload = _bundle_manifest_status(
        root,
        artifacts=artifacts,
        check_bundle_hashes=bool(check_bundle_hashes),
    )
    (
        has_bundle_operator_contract,
        has_bundle_operator_contract_consistent,
        has_bundle_operator_contract_digests_valid,
        bundle_operator_contract_error_present,
        bundle_operator_contract_digest_error_present,
    ) = _bundle_operator_contract_flags(
        root,
        bundle_manifest_payload=bundle_manifest_payload,
    )

    weights_audit = _evaluate_bundle_weights_audit(
        root / "deploy_bundle",
        check_hashes=bool(check_bundle_hashes),
    )
    bundle_weights_valid = (weights_audit["valid"] is True) or (weights_audit["present"] is False)
    dataset_readiness = _report_dataset_readiness(root)

    report_present = bool(artifacts["report"]["present"])
    core_complete = len(missing_required) == 0
    audited_complete = (
        core_complete
        and bool(artifacts["infer_config"]["present"])
        and bool(artifacts["calibration_card"]["present"])
        and calibration_valid is True
        and calibration_audit["matching_threshold"] is not False
        and calibration_audit["matching_split_fingerprint"] is not False
        and operator_contract_audit["matching_payload"] is not False
    )
    deployable_complete = (
        audited_complete
        and bool(artifacts["deploy_bundle_manifest"]["present"])
        and bundle_manifest_payload["valid"] is True
        and bool(bundle_weights_valid)
    )
    status, score = _quality_status_and_score(
        report_present=report_present,
        core_complete=core_complete,
        audited_complete=audited_complete,
        deployable_complete=deployable_complete,
    )

    warnings = list(calibration_audit["warnings"])
    warnings.extend(str(item) for item in operator_contract_audit["warnings"])
    trust_summary = _build_trust_summary(
        artifacts=artifacts,
        missing_required=missing_required,
        core_complete=bool(core_complete),
        audited_complete=bool(audited_complete),
        deployable_complete=bool(deployable_complete),
        warnings=warnings,
        calibration_audit=calibration_audit,
        operator_contract_audit=operator_contract_audit,
        bundle_manifest=bundle_manifest_payload,
        weights_audit=weights_audit,
        has_bundle_operator_contract=has_bundle_operator_contract,
        has_bundle_operator_contract_consistent=has_bundle_operator_contract_consistent,
        has_bundle_operator_contract_digests_valid=has_bundle_operator_contract_digests_valid,
        has_bundle_operator_contract_error=bundle_operator_contract_error_present,
        has_bundle_operator_contract_digest_error=bundle_operator_contract_digest_error_present,
    )

    return {
        "run_dir": str(root),
        "status": status,
        "score": float(score),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "core_complete": bool(core_complete),
        "audited_complete": bool(audited_complete),
        "deployable_complete": bool(deployable_complete),
        "warnings": warnings,
        "artifacts": artifacts,
        "calibration_audit": calibration_audit,
        "operator_contract_audit": operator_contract_audit,
        "bundle_manifest": bundle_manifest_payload,
        "weights_audit": weights_audit,
        "dataset_readiness": dataset_readiness,
        "trust_summary": trust_summary,
    }


__all__ = ["evaluate_run_quality"]
