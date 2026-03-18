from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyimgano.reporting.calibration_card import validate_calibration_card_payload
from pyimgano.reporting.deploy_bundle import validate_deploy_bundle_manifest
from pyimgano.weights.manifest import validate_weights_manifest_file
from pyimgano.weights.model_card import validate_model_card_file


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


def _validation_payload(rel_path: str) -> dict[str, Any]:
    return {
        "path": rel_path,
        "present": False,
        "valid": None,
        "errors": [],
        "warnings": [],
    }


def _extract_threshold_value(payload: Mapping[str, Any]) -> float | None:
    threshold = payload.get("threshold", None)
    if isinstance(threshold, (int, float)):
        return float(threshold)

    image_threshold = payload.get("image_threshold", None)
    if isinstance(image_threshold, Mapping):
        nested = image_threshold.get("threshold", None)
        if isinstance(nested, (int, float)):
            return float(nested)

    return None


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

    infer_payload: dict[str, Any] | None = None
    if infer_config_present:
        infer_path = root / "artifacts" / "infer_config.json"
        try:
            infer_payload = _load_json_dict(infer_path)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            warnings.append(f"Failed to inspect infer_config.json for calibration audit: {exc}")

    calibration_payload: dict[str, Any] | None = None
    if calibration_present and calibration_valid is not False:
        calibration_path = root / "artifacts" / "calibration_card.json"
        try:
            calibration_payload = _load_json_dict(calibration_path)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            warnings.append(f"Failed to inspect calibration_card.json for calibration audit: {exc}")

    if isinstance(calibration_payload, Mapping):
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

    if (
        calibration_valid is True
        and isinstance(infer_payload, Mapping)
        and isinstance(calibration_payload, Mapping)
    ):
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
            warnings.append("Calibration audit warning: calibration_card.json is missing split_fingerprint metadata.")

        if isinstance(infer_payload.get("prediction", None), Mapping) and not bool(
            payload["has_prediction_policy"]
        ):
            warnings.append(
                "Calibration audit warning: calibration_card.json is missing prediction_policy metadata."
            )
        if not bool(payload["has_threshold_context"]):
            warnings.append(
                "Calibration audit warning: calibration_card.json is missing threshold_context metadata."
            )

    payload["warnings"] = warnings
    return payload


def _evaluate_bundle_weights_audit(
    bundle_dir: str | Path,
    *,
    check_hashes: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    payload = {
        "present": False,
        "valid": None,
        "cross_checked": False,
        "model_card": _validation_payload("model_card.json"),
        "weights_manifest": _validation_payload("weights_manifest.json"),
    }

    manifest_path = bundle_root / "weights_manifest.json"
    manifest_ok = False
    if manifest_path.is_file():
        payload["present"] = True
        payload["weights_manifest"]["present"] = True
        report = validate_weights_manifest_file(
            manifest_path=manifest_path,
            check_files=True,
            check_hashes=bool(check_hashes),
        )
        payload["weights_manifest"]["valid"] = bool(report.ok)
        payload["weights_manifest"]["errors"] = list(report.errors)
        payload["weights_manifest"]["warnings"] = list(report.warnings)
        manifest_ok = bool(report.ok)

    model_card_path = bundle_root / "model_card.json"
    if model_card_path.is_file():
        payload["present"] = True
        payload["model_card"]["present"] = True
        report = validate_model_card_file(
            model_card_path,
            manifest_path=(manifest_path if manifest_ok else None),
            check_files=True,
            check_hashes=bool(check_hashes),
        )
        payload["model_card"]["valid"] = bool(report.ok)
        payload["model_card"]["errors"] = list(report.errors)
        payload["model_card"]["warnings"] = list(report.warnings)
        payload["cross_checked"] = bool(manifest_ok)

    if bool(payload["present"]):
        valid = True
        for key in ("model_card", "weights_manifest"):
            artifact_payload = payload[key]
            if bool(artifact_payload["present"]) and artifact_payload["valid"] is False:
                valid = False
        payload["valid"] = bool(valid)

    return payload


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
    has_bundle_operator_contract_error: bool,
) -> dict[str, Any]:
    status_reasons: list[str] = []
    degraded_by: list[str] = []
    trust_signals = {
        "has_core_artifacts": bool(core_complete),
        "has_infer_config": bool(artifacts.get("infer_config", {}).get("present")),
        "has_operator_contract": bool(artifacts.get("operator_contract", {}).get("present")),
        "has_operator_contract_consistent": bool(
            operator_contract_audit.get("matching_payload") is True
        ),
        "has_bundle_operator_contract": bool(has_bundle_operator_contract),
        "has_bundle_operator_contract_consistent": bool(has_bundle_operator_contract_consistent),
        "has_calibration_card": bool(artifacts.get("calibration_card", {}).get("present")),
        "has_threshold_context": bool(calibration_audit.get("has_threshold_context")),
        "has_split_fingerprint": bool(calibration_audit.get("has_split_fingerprint")),
        "has_prediction_policy": bool(calibration_audit.get("has_prediction_policy")),
        "has_deploy_bundle_manifest": bool(bundle_manifest.get("present")),
        "has_valid_deploy_bundle_manifest": bool(bundle_manifest.get("valid") is True),
        "has_bundle_weights_audit": bool(weights_audit.get("present")),
        "has_valid_bundle_weights_audit": bool(weights_audit.get("valid") is True),
    }

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
    if missing_required:
        degraded_by.append("missing_required_artifacts")

    for warning in warnings:
        text = str(warning).lower()
        if "threshold_context" in text and "missing_threshold_context" not in degraded_by:
            degraded_by.append("missing_threshold_context")
        if "split_fingerprint" in text and "missing_split_fingerprint" not in degraded_by:
            degraded_by.append("missing_split_fingerprint")
        if "prediction_policy" in text and "missing_prediction_policy" not in degraded_by:
            degraded_by.append("missing_prediction_policy")
        if "threshold mismatch" in text and "threshold_mismatch" not in degraded_by:
            degraded_by.append("threshold_mismatch")
        if "operator contract" in text and "operator_contract_mismatch" not in degraded_by:
            degraded_by.append("operator_contract_mismatch")

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

    report_present = bool(artifacts.get("report", {}).get("present"))
    if not report_present:
        status = "broken"
    elif not bool(core_complete):
        status = "partial"
    elif bool(audited_complete) and not degraded_by:
        status = "trust-signaled"
    else:
        status = "partial"

    return {
        "status": status,
        "trust_signals": trust_signals,
        "status_reasons": list(dict.fromkeys(str(item) for item in status_reasons)),
        "degraded_by": list(dict.fromkeys(str(item) for item in degraded_by)),
        "audit_refs": _build_audit_refs(artifacts=artifacts, weights_audit=weights_audit),
    }


def evaluate_run_quality(
    run_dir: str | Path,
    *,
    check_bundle_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)

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
        if not present:
            if spec.required:
                missing_required.append(spec.rel_path)
            else:
                missing_optional.append(spec.rel_path)

    calibration_valid = None
    if bool(artifacts["calibration_card"]["present"]):
        calibration_path = root / "artifacts" / "calibration_card.json"
        try:
            calibration_payload = _load_json_dict(calibration_path)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            artifacts["calibration_card"]["valid"] = False
            artifacts["calibration_card"]["errors"] = [str(exc)]
            calibration_valid = False
        else:
            errors = validate_calibration_card_payload(calibration_payload)
            artifacts["calibration_card"]["valid"] = len(errors) == 0
            artifacts["calibration_card"]["errors"] = list(errors)
            calibration_valid = len(errors) == 0

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

    bundle_manifest_payload = {
        "present": bool(artifacts["deploy_bundle_manifest"]["present"]),
        "valid": None,
        "errors": [],
    }
    if bool(artifacts["deploy_bundle_manifest"]["present"]):
        manifest_path = root / "deploy_bundle" / "bundle_manifest.json"
        try:
            manifest = _load_json_dict(manifest_path)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            bundle_manifest_payload["valid"] = False
            bundle_manifest_payload["errors"] = [str(exc)]
        else:
            errors = validate_deploy_bundle_manifest(
                manifest,
                bundle_dir=root / "deploy_bundle",
                check_hashes=bool(check_bundle_hashes),
            )
            bundle_manifest_payload["valid"] = len(errors) == 0
            bundle_manifest_payload["errors"] = list(errors)

    has_bundle_operator_contract = bool((root / "deploy_bundle" / "operator_contract.json").is_file())
    bundle_operator_contract_error_present = any(
        "operator_contract" in str(item).lower() or "operator contract" in str(item).lower()
        for item in bundle_manifest_payload["errors"]
    )
    has_bundle_operator_contract_consistent = (
        bool(has_bundle_operator_contract)
        and bool(bundle_manifest_payload.get("present"))
        and not bool(bundle_operator_contract_error_present)
    )

    weights_audit = _evaluate_bundle_weights_audit(
        root / "deploy_bundle",
        check_hashes=bool(check_bundle_hashes),
    )
    bundle_weights_valid = (weights_audit["valid"] is True) or (weights_audit["present"] is False)

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

    if deployable_complete:
        status = "deployable"
        score = 1.0
    elif audited_complete:
        status = "audited"
        score = 0.75
    elif core_complete:
        status = "reproducible"
        score = 0.5
    elif report_present:
        status = "partial"
        score = 0.25
    else:
        status = "broken"
        score = 0.0

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
        has_bundle_operator_contract_error=bundle_operator_contract_error_present,
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
        "trust_summary": trust_summary,
    }


__all__ = ["evaluate_run_quality"]
