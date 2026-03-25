from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pyimgano.weights.manifest import validate_weights_manifest_file
from pyimgano.weights.model_card import validate_model_card_file


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def _resolve_metadata_path(path: str | Path) -> Path:
    candidate = Path(path).resolve(strict=False)
    if candidate.is_dir():
        return (candidate / "leaderboard_metadata.json").resolve(strict=False)
    return candidate


def _resolve_exported_path(root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    root_resolved = root.resolve(strict=False)
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.relative_to(root_resolved)
        except ValueError:
            return None
    if path.is_absolute() or ".." in path.parts:
        return None
    candidate = (root_resolved / path).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _display_exported_path(root: Path, raw: Any) -> str | None:
    resolved = _resolve_exported_path(root, raw)
    if resolved is None:
        return None
    try:
        return str(resolved.relative_to(root.resolve(strict=False)))
    except Exception:
        return str(resolved)


def _validation_payload(rel_path: str) -> dict[str, Any]:
    return {
        "path": rel_path,
        "present": False,
        "valid": None,
        "errors": [],
        "warnings": [],
    }


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _append_missing_required(missing_required: list[str], item: str) -> None:
    if item not in missing_required:
        missing_required.append(item)


def _split_fingerprint_sha256(split_fingerprint: Any) -> str | None:
    if not isinstance(split_fingerprint, dict):
        return None
    return _nonempty_str(split_fingerprint.get("sha256"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_exported_digest_keys(exported_files: Any) -> list[str]:
    if not isinstance(exported_files, dict):
        return []
    return [str(key) for key in exported_files if str(key) != "leaderboard_metadata_json"]


def _evaluate_declared_weight_artifacts(
    root: Path,
    exported_files: Any,
) -> tuple[list[str], dict[str, Any]]:
    payload = {
        "present": False,
        "valid": None,
        "cross_checked": False,
        "model_card": _validation_payload("model_card_json"),
        "weights_manifest": _validation_payload("weights_manifest_json"),
    }
    invalid_declared: list[str] = []
    if not isinstance(exported_files, dict):
        return invalid_declared, payload

    manifest_path = _resolve_exported_path(root, exported_files.get("weights_manifest_json"))
    manifest_ok = False
    if manifest_path is not None and manifest_path.is_file():
        payload["present"] = True
        payload["weights_manifest"]["present"] = True
        payload["weights_manifest"]["path"] = str(manifest_path)
        report = validate_weights_manifest_file(
            manifest_path=manifest_path,
            check_files=False,
            check_hashes=False,
        )
        payload["weights_manifest"]["valid"] = bool(report.ok)
        payload["weights_manifest"]["errors"] = list(report.errors)
        payload["weights_manifest"]["warnings"] = list(report.warnings)
        if not report.ok:
            invalid_declared.append("weights_manifest_json")
        manifest_ok = bool(report.ok)

    model_card_path = _resolve_exported_path(root, exported_files.get("model_card_json"))
    if model_card_path is not None and model_card_path.is_file():
        payload["present"] = True
        payload["model_card"]["present"] = True
        payload["model_card"]["path"] = str(model_card_path)
        report = validate_model_card_file(
            model_card_path,
            manifest_path=(manifest_path if manifest_ok else None),
            check_files=False,
            check_hashes=False,
        )
        payload["model_card"]["valid"] = bool(report.ok)
        payload["model_card"]["errors"] = list(report.errors)
        payload["model_card"]["warnings"] = list(report.warnings)
        if not report.ok:
            invalid_declared.append("model_card_json")
        payload["cross_checked"] = bool(manifest_ok)

    if bool(payload["present"]):
        valid = True
        for key in ("model_card", "weights_manifest"):
            artifact_payload = payload[key]
            if bool(artifact_payload["present"]) and artifact_payload["valid"] is False:
                valid = False
        payload["valid"] = bool(valid)

    return invalid_declared, payload


def evaluate_publication_quality(path: str | Path) -> dict[str, Any]:
    metadata_path = _resolve_metadata_path(path)
    root = metadata_path.parent
    if not metadata_path.is_file():
        return {
            "root": str(root),
            "metadata_path": str(metadata_path),
            "status": "broken",
            "publication_ready": False,
            "missing_required": ["leaderboard_metadata.json"],
            "exported_files_present": {},
            "exported_file_digests": {},
            "artifact_quality": None,
            "invalid_declared": [],
            "asset_audit": {
                "present": False,
                "valid": None,
                "cross_checked": False,
                "model_card": _validation_payload("model_card_json"),
                "weights_manifest": _validation_payload("weights_manifest_json"),
            },
        }

    metadata = _load_json_dict(metadata_path)
    artifact_quality = metadata.get("artifact_quality", None)
    benchmark_config = metadata.get("benchmark_config", None)
    declared_missing = []
    if isinstance(artifact_quality, dict):
        declared_missing = [
            str(item)
            for item in artifact_quality.get("missing_required", [])
            if isinstance(item, str) and item
        ]

    exported_files = metadata.get("exported_files", None)
    exported_files_present: dict[str, bool] = {}
    audit_refs: dict[str, str] = {"leaderboard_metadata_json": "leaderboard_metadata.json"}
    missing_required = list(declared_missing)
    resolved_exported_paths: dict[str, Path] = {}
    if isinstance(exported_files, dict):
        for key, raw_path in exported_files.items():
            resolved = _resolve_exported_path(root, raw_path)
            present = bool(resolved is not None and resolved.is_file())
            exported_files_present[str(key)] = present
            if present and resolved is not None:
                resolved_exported_paths[str(key)] = resolved
            display_path = _display_exported_path(root, raw_path)
            if display_path is not None and present:
                audit_refs[str(key)] = str(display_path)
            if not present and str(key) != "leaderboard_metadata_json" and str(key) not in missing_required:
                missing_required.append(str(key))
    if not any(
        key.startswith("leaderboard_") and key != "leaderboard_metadata_json"
        for key in exported_files_present
    ):
        _append_missing_required(missing_required, "leaderboard_table")

    required_exported_digest_keys = _required_exported_digest_keys(exported_files)
    metadata_exported_file_digests_raw = metadata.get("exported_file_digests", None)
    metadata_exported_file_digests = (
        dict(metadata_exported_file_digests_raw)
        if isinstance(metadata_exported_file_digests_raw, dict)
        else {}
    )
    exported_file_digests: dict[str, str] = {}
    has_exported_file_digests = bool(required_exported_digest_keys)
    for key in required_exported_digest_keys:
        digest = _nonempty_str(metadata_exported_file_digests.get(key))
        if digest is None:
            _append_missing_required(missing_required, f"exported_file_digests.{key}")
            has_exported_file_digests = False
            continue
        resolved_exported = resolved_exported_paths.get(key)
        if resolved_exported is None:
            has_exported_file_digests = False
            continue
        actual_digest = _file_sha256(resolved_exported)
        if digest != actual_digest:
            _append_missing_required(missing_required, f"{key}_sha256_mismatch")
            has_exported_file_digests = False
            continue
        exported_file_digests[key] = digest

    metadata_audit_refs_raw = metadata.get("audit_refs", None)
    metadata_audit_refs = (
        dict(metadata_audit_refs_raw) if isinstance(metadata_audit_refs_raw, dict) else {}
    )
    has_run_artifact_refs = True
    resolved_run_artifact_paths: dict[str, Path] = {}
    for key in ("report_json", "config_json", "environment_json"):
        raw_ref = _nonempty_str(metadata_audit_refs.get(key))
        if raw_ref is None:
            _append_missing_required(missing_required, f"audit_refs.{key}")
            has_run_artifact_refs = False
            continue
        resolved_ref = _resolve_exported_path(root, raw_ref)
        if resolved_ref is None or not resolved_ref.is_file():
            _append_missing_required(missing_required, key)
            has_run_artifact_refs = False
            continue
        resolved_run_artifact_paths[key] = resolved_ref
        display_ref = _display_exported_path(root, raw_ref)
        if display_ref is not None:
            audit_refs[key] = str(display_ref)

    metadata_audit_digests_raw = metadata.get("audit_digests", None)
    metadata_audit_digests = (
        dict(metadata_audit_digests_raw) if isinstance(metadata_audit_digests_raw, dict) else {}
    )
    audit_digests: dict[str, str] = {}
    has_run_artifact_digests = True
    for key in ("report_json", "config_json", "environment_json"):
        digest = _nonempty_str(metadata_audit_digests.get(key))
        if digest is None:
            _append_missing_required(missing_required, f"audit_digests.{key}")
            has_run_artifact_digests = False
            continue
        resolved_ref = resolved_run_artifact_paths.get(key)
        if resolved_ref is None:
            has_run_artifact_digests = False
            continue
        actual_digest = _file_sha256(resolved_ref)
        if digest != actual_digest:
            _append_missing_required(missing_required, f"{key}_sha256_mismatch")
            has_run_artifact_digests = False
            continue
        audit_digests[key] = digest

    benchmark_config_source = None
    benchmark_config_sha256 = None
    benchmark_config_trust = None
    if isinstance(benchmark_config, dict):
        raw_source = _nonempty_str(benchmark_config.get("source", None))
        if raw_source is not None:
            benchmark_config_source = raw_source
            audit_refs["benchmark_config_source"] = str(raw_source)
        raw_sha256 = _nonempty_str(benchmark_config.get("sha256", None))
        if raw_sha256 is not None:
            benchmark_config_sha256 = raw_sha256
        raw_trust = benchmark_config.get("trust_summary", None)
        if isinstance(raw_trust, dict):
            benchmark_config_trust = raw_trust
    else:
        _append_missing_required(missing_required, "benchmark_config")

    if isinstance(benchmark_config, dict) and benchmark_config_source is None:
        _append_missing_required(missing_required, "benchmark_config.source")
    if isinstance(benchmark_config, dict) and benchmark_config_sha256 is None:
        _append_missing_required(missing_required, "benchmark_config.sha256")

    environment_fingerprint_sha256 = _nonempty_str(metadata.get("environment_fingerprint_sha256"))
    if environment_fingerprint_sha256 is None:
        _append_missing_required(missing_required, "environment_fingerprint_sha256")

    split_fingerprint_sha256 = _split_fingerprint_sha256(metadata.get("split_fingerprint"))
    if split_fingerprint_sha256 is None:
        _append_missing_required(missing_required, "split_fingerprint.sha256")

    has_evaluation_contract = isinstance(metadata.get("evaluation_contract"), dict)
    if not has_evaluation_contract:
        _append_missing_required(missing_required, "evaluation_contract")

    has_benchmark_citation = isinstance(metadata.get("citation"), dict)
    if not has_benchmark_citation:
        _append_missing_required(missing_required, "citation")

    invalid_declared, asset_audit = _evaluate_declared_weight_artifacts(root, exported_files)
    has_official_benchmark_config = bool(
        isinstance(benchmark_config, dict) and benchmark_config.get("official")
    )
    has_benchmark_provenance = bool(
        isinstance(benchmark_config, dict)
        and benchmark_config.get("official")
        and benchmark_config_source
        and benchmark_config_sha256
    )
    trust_signals = {
        "has_official_benchmark_config": has_official_benchmark_config,
        "has_evaluation_contract": has_evaluation_contract,
        "has_benchmark_citation": has_benchmark_citation,
        "has_cross_checked_assets": bool(asset_audit.get("cross_checked")),
        "has_benchmark_provenance": has_benchmark_provenance,
        "has_benchmark_config_ref": bool(benchmark_config_source),
        "has_run_artifact_refs": bool(has_run_artifact_refs),
        "has_run_artifact_digests": bool(has_run_artifact_digests),
        "has_exported_file_digests": bool(has_exported_file_digests),
    }
    if isinstance(benchmark_config_trust, dict):
        trust_signals["has_trust_signaled_benchmark_config"] = (
            str(benchmark_config_trust.get("status", "")) == "trust-signaled"
        )

    artifact_quality_payload = dict(artifact_quality) if isinstance(artifact_quality, dict) else {}
    artifact_quality_payload["required_files_present"] = len(missing_required) == 0
    artifact_quality_payload["missing_required"] = list(missing_required)
    artifact_quality_payload["has_official_benchmark_config"] = has_official_benchmark_config
    artifact_quality_payload["has_environment_fingerprint"] = bool(environment_fingerprint_sha256)
    artifact_quality_payload["has_split_fingerprint"] = bool(split_fingerprint_sha256)
    artifact_quality_payload["has_evaluation_contract"] = has_evaluation_contract
    artifact_quality_payload["has_benchmark_citation"] = has_benchmark_citation
    artifact_quality_payload["has_benchmark_provenance"] = has_benchmark_provenance
    artifact_quality_payload["has_run_artifact_refs"] = bool(has_run_artifact_refs)
    artifact_quality_payload["has_run_artifact_digests"] = bool(has_run_artifact_digests)
    artifact_quality_payload["has_exported_file_digests"] = bool(has_exported_file_digests)
    if "has_trust_signaled_benchmark_config" in trust_signals:
        artifact_quality_payload["has_trust_signaled_benchmark_config"] = bool(
            trust_signals["has_trust_signaled_benchmark_config"]
        )

    publication_ready = bool(
        metadata.get("publication_ready")
        and artifact_quality_payload["required_files_present"]
        and artifact_quality_payload["has_official_benchmark_config"]
        and artifact_quality_payload["has_evaluation_contract"]
        and artifact_quality_payload["has_benchmark_citation"]
        and artifact_quality_payload["has_benchmark_provenance"]
        and artifact_quality_payload["has_run_artifact_refs"]
        and artifact_quality_payload["has_run_artifact_digests"]
        and artifact_quality_payload["has_exported_file_digests"]
        and not invalid_declared
    )
    if publication_ready:
        status = "ready"
    else:
        status = "partial"

    return {
        "root": str(root),
        "metadata_path": str(metadata_path),
        "status": status,
        "publication_ready": bool(publication_ready),
        "missing_required": missing_required,
        "exported_files_present": exported_files_present,
        "exported_file_digests": exported_file_digests,
        "artifact_quality": artifact_quality_payload,
        "invalid_declared": invalid_declared,
        "asset_audit": asset_audit,
        "trust_signals": trust_signals,
        "audit_refs": audit_refs,
        "audit_digests": audit_digests,
    }


__all__ = ["evaluate_publication_quality"]
