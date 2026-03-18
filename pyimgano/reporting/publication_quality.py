from __future__ import annotations

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
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "leaderboard_metadata.json"
    return candidate


def _resolve_exported_path(root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return root / path


def _display_exported_path(root: Path, raw: Any) -> str | None:
    resolved = _resolve_exported_path(root, raw)
    if resolved is None:
        return None
    try:
        return str(resolved.relative_to(root))
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
    if isinstance(exported_files, dict):
        for key, raw_path in exported_files.items():
            resolved = _resolve_exported_path(root, raw_path)
            present = bool(resolved is not None and resolved.is_file())
            exported_files_present[str(key)] = present
            display_path = _display_exported_path(root, raw_path)
            if display_path is not None and present:
                audit_refs[str(key)] = str(display_path)
            if not present and str(key) != "leaderboard_metadata_json" and str(key) not in missing_required:
                missing_required.append(str(key))

    benchmark_config_source = None
    benchmark_config_sha256 = None
    benchmark_config_trust = None
    if isinstance(benchmark_config, dict):
        raw_source = benchmark_config.get("source", None)
        if isinstance(raw_source, str) and raw_source:
            benchmark_config_source = raw_source
            audit_refs["benchmark_config_source"] = str(raw_source)
        raw_sha256 = benchmark_config.get("sha256", None)
        if isinstance(raw_sha256, str) and raw_sha256:
            benchmark_config_sha256 = raw_sha256
        raw_trust = benchmark_config.get("trust_summary", None)
        if isinstance(raw_trust, dict):
            benchmark_config_trust = raw_trust

    invalid_declared, asset_audit = _evaluate_declared_weight_artifacts(root, exported_files)
    trust_signals = {
        "has_official_benchmark_config": bool(
            isinstance(artifact_quality, dict) and artifact_quality.get("has_official_benchmark_config")
        ),
        "has_evaluation_contract": isinstance(metadata.get("evaluation_contract"), dict),
        "has_benchmark_citation": isinstance(metadata.get("citation"), dict),
        "has_cross_checked_assets": bool(asset_audit.get("cross_checked")),
        "has_benchmark_provenance": bool(
            isinstance(benchmark_config, dict)
            and benchmark_config.get("official")
            and benchmark_config_source
            and benchmark_config_sha256
        ),
        "has_benchmark_config_ref": bool(benchmark_config_source),
    }
    if isinstance(benchmark_config_trust, dict):
        trust_signals["has_trust_signaled_benchmark_config"] = (
            str(benchmark_config_trust.get("status", "")) == "trust-signaled"
        )

    publication_ready = bool(metadata.get("publication_ready"))
    if publication_ready and not missing_required and not invalid_declared:
        status = "ready"
    else:
        status = "partial"

    return {
        "root": str(root),
        "metadata_path": str(metadata_path),
        "status": status,
        "publication_ready": bool(publication_ready and not missing_required and not invalid_declared),
        "missing_required": missing_required,
        "exported_files_present": exported_files_present,
        "artifact_quality": artifact_quality,
        "invalid_declared": invalid_declared,
        "asset_audit": asset_audit,
        "trust_signals": trust_signals,
        "audit_refs": audit_refs,
    }


__all__ = ["evaluate_publication_quality"]
