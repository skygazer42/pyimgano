from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pyimgano.weights.manifest import validate_weights_manifest_file
from pyimgano.weights.model_card import validate_model_card_file

_MODEL_CARD_JSON = "model_card.json"
_WEIGHTS_MANIFEST_JSON = "weights_manifest.json"


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _has_runtime(entry: Mapping[str, Any]) -> bool:
    runtime = _nonempty_str(entry.get("runtime", None))
    if runtime is not None:
        return True
    runtimes = entry.get("runtimes", None)
    if not isinstance(runtimes, list):
        return False
    return any(_nonempty_str(item) is not None for item in runtimes)


def _validation_payload(rel_path: str) -> dict[str, Any]:
    return {
        "path": rel_path,
        "present": False,
        "valid": None,
        "errors": [],
        "warnings": [],
    }


def _build_bundle_trust_summary(
    *,
    manifest_present: bool,
    manifest_valid: bool | None,
    manifest_entries: list[dict[str, Any]],
    model_card_present: bool,
    model_card_valid: bool | None,
    model_card_normalized: Mapping[str, Any],
    model_card_assets: Mapping[str, Any],
    cross_checked: bool,
    check_hashes: bool,
) -> dict[str, Any]:
    weights = (
        dict(model_card_normalized.get("weights", {}))
        if isinstance(model_card_normalized.get("weights", {}), Mapping)
        else {}
    )
    deployment = (
        dict(model_card_normalized.get("deployment", {}))
        if isinstance(model_card_normalized.get("deployment", {}), Mapping)
        else {}
    )
    manifest_asset = (
        dict(model_card_assets.get("manifest", {}))
        if isinstance(model_card_assets.get("manifest", {}), Mapping)
        else {}
    )

    trust_signals = {
        "file_refs_checked": True,
        "hashes_checked": bool(check_hashes),
        "has_model_card": bool(model_card_present),
        "model_card_valid": bool(model_card_valid is True),
        "has_weights_manifest": bool(manifest_present),
        "weights_manifest_valid": bool(manifest_valid is True),
        "has_cross_checked_manifest": bool(cross_checked),
        "has_weights_sha256": _nonempty_str(weights.get("sha256", None)) is not None,
        "has_weights_source": _nonempty_str(weights.get("source", None)) is not None,
        "has_weights_license": _nonempty_str(weights.get("license", None)) is not None,
        "has_deployment_runtime": _nonempty_str(deployment.get("runtime", None)) is not None,
        "has_manifest_link": (
            _nonempty_str(weights.get("manifest_entry", None)) is not None
            or _nonempty_str(manifest_asset.get("matched_entry", None)) is not None
        ),
        "manifest_has_entries": bool(len(manifest_entries) > 0),
        "manifest_all_entries_have_sha256": bool(
            len(manifest_entries) > 0
            and all(_nonempty_str(item.get("sha256", None)) is not None for item in manifest_entries)
        ),
        "manifest_all_entries_have_source": bool(
            len(manifest_entries) > 0
            and all(_nonempty_str(item.get("source", None)) is not None for item in manifest_entries)
        ),
        "manifest_all_entries_have_license": bool(
            len(manifest_entries) > 0
            and all(_nonempty_str(item.get("license", None)) is not None for item in manifest_entries)
        ),
        "manifest_all_entries_have_runtime": bool(
            len(manifest_entries) > 0 and all(_has_runtime(item) for item in manifest_entries)
        ),
    }

    degraded_by: list[str] = []
    if not bool(model_card_present):
        degraded_by.append("missing_model_card")
    if not bool(manifest_present):
        degraded_by.append("missing_weights_manifest")
    if bool(model_card_present) and model_card_valid is False:
        degraded_by.append("invalid_model_card")
    if bool(manifest_present) and manifest_valid is False:
        degraded_by.append("invalid_weights_manifest")

    if bool(model_card_present):
        if not trust_signals["has_weights_sha256"]:
            degraded_by.append("missing_weights_sha256")
        if not trust_signals["has_weights_source"]:
            degraded_by.append("missing_weights_source")
        if not trust_signals["has_weights_license"]:
            degraded_by.append("missing_weights_license")
        if not trust_signals["has_deployment_runtime"]:
            degraded_by.append("missing_deployment_runtime")
        if not trust_signals["has_manifest_link"]:
            degraded_by.append("missing_manifest_link")
        if not trust_signals["has_cross_checked_manifest"]:
            degraded_by.append("missing_cross_checked_manifest")

    if bool(manifest_present):
        if not trust_signals["manifest_has_entries"]:
            degraded_by.append("manifest_missing_entries")
        if not trust_signals["manifest_all_entries_have_sha256"]:
            degraded_by.append("manifest_missing_sha256")
        if not trust_signals["manifest_all_entries_have_source"]:
            degraded_by.append("manifest_missing_source")
        if not trust_signals["manifest_all_entries_have_license"]:
            degraded_by.append("manifest_missing_license")
        if not trust_signals["manifest_all_entries_have_runtime"]:
            degraded_by.append("manifest_missing_runtime")

    status = "trust-signaled" if not degraded_by else "partial"
    return {
        "status": status,
        "trust_signals": trust_signals,
        "degraded_by": list(dict.fromkeys(str(item) for item in degraded_by)),
        "audit_refs": {
            key: value
            for key, value in {
                "model_card_json": _MODEL_CARD_JSON if model_card_present else None,
                "weights_manifest_json": _WEIGHTS_MANIFEST_JSON if manifest_present else None,
            }.items()
            if value is not None
        },
    }


def evaluate_bundle_weights_audit(
    bundle_dir: str | Path,
    *,
    check_hashes: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    if not bundle_root.exists():
        raise FileNotFoundError(f"Deploy bundle not found: {bundle_root}")
    if not bundle_root.is_dir():
        raise NotADirectoryError(f"Deploy bundle must be a directory: {bundle_root}")

    payload = {
        "bundle_dir": str(bundle_root),
        "present": False,
        "valid": None,
        "cross_checked": False,
        "missing_required": [],
        "warnings": [],
        "errors": [],
        "status": "partial",
        "ready": False,
        "model_card": _validation_payload(_MODEL_CARD_JSON),
        "weights_manifest": _validation_payload(_WEIGHTS_MANIFEST_JSON),
        "trust_summary": {},
    }

    manifest_path = bundle_root / _WEIGHTS_MANIFEST_JSON
    model_card_path = bundle_root / _MODEL_CARD_JSON

    manifest_present = manifest_path.is_file()
    model_card_present = model_card_path.is_file()
    payload["present"] = bool(manifest_present or model_card_present)

    manifest_report = None
    manifest_valid = None
    manifest_entries: list[dict[str, Any]] = []
    if manifest_present:
        payload["weights_manifest"]["present"] = True
        manifest_report = validate_weights_manifest_file(
            manifest_path=manifest_path,
            check_files=True,
            check_hashes=bool(check_hashes),
        )
        manifest_valid = bool(manifest_report.ok)
        manifest_entries = [dict(item) for item in manifest_report.entries]
        payload["weights_manifest"]["valid"] = bool(manifest_valid)
        payload["weights_manifest"]["errors"] = list(manifest_report.errors)
        payload["weights_manifest"]["warnings"] = list(manifest_report.warnings)
        payload["warnings"].extend(
            f"{_WEIGHTS_MANIFEST_JSON}: {item}" for item in payload["weights_manifest"]["warnings"]
        )
        payload["errors"].extend(
            f"{_WEIGHTS_MANIFEST_JSON}: {item}" for item in payload["weights_manifest"]["errors"]
        )
    else:
        payload["missing_required"].append(_WEIGHTS_MANIFEST_JSON)

    model_card_report = None
    model_card_valid = None
    model_card_normalized: Mapping[str, Any] = {}
    model_card_assets: Mapping[str, Any] = {}
    if model_card_present:
        payload["model_card"]["present"] = True
        model_card_report = validate_model_card_file(
            model_card_path,
            manifest_path=(manifest_path if manifest_valid is True else None),
            check_files=True,
            check_hashes=bool(check_hashes),
        )
        model_card_valid = bool(model_card_report.ok)
        model_card_normalized = dict(model_card_report.normalized)
        model_card_assets = dict(model_card_report.assets)
        payload["model_card"]["valid"] = bool(model_card_valid)
        payload["model_card"]["errors"] = list(model_card_report.errors)
        payload["model_card"]["warnings"] = list(model_card_report.warnings)
        payload["cross_checked"] = bool(
            isinstance(model_card_assets.get("manifest", None), Mapping)
            and model_card_assets["manifest"].get("ok", None) is True
        )
        payload["warnings"].extend(
            f"{_MODEL_CARD_JSON}: {item}" for item in payload["model_card"]["warnings"]
        )
        payload["errors"].extend(
            f"{_MODEL_CARD_JSON}: {item}" for item in payload["model_card"]["errors"]
        )
    else:
        payload["missing_required"].append(_MODEL_CARD_JSON)

    if bool(payload["present"]):
        payload["valid"] = bool(
            (not bool(manifest_present) or manifest_valid is True)
            and (not bool(model_card_present) or model_card_valid is True)
        )

    trust_summary = _build_bundle_trust_summary(
        manifest_present=bool(manifest_present),
        manifest_valid=manifest_valid,
        manifest_entries=manifest_entries,
        model_card_present=bool(model_card_present),
        model_card_valid=model_card_valid,
        model_card_normalized=model_card_normalized,
        model_card_assets=model_card_assets,
        cross_checked=bool(payload["cross_checked"]),
        check_hashes=bool(check_hashes),
    )
    payload["trust_summary"] = trust_summary
    payload["ready"] = bool(trust_summary.get("status") == "trust-signaled")
    payload["status"] = "ready" if bool(payload["ready"]) else "partial"
    payload["missing_required"] = list(dict.fromkeys(str(item) for item in payload["missing_required"]))
    payload["warnings"] = list(dict.fromkeys(str(item) for item in payload["warnings"]))
    payload["errors"] = list(dict.fromkeys(str(item) for item in payload["errors"]))
    return payload


__all__ = ["evaluate_bundle_weights_audit"]
