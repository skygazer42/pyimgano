from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.reporting.deploy_bundle_validation_helpers import (
    append_operator_contract_presence_errors as _append_operator_contract_presence_errors_helper,
)
from pyimgano.reporting.deploy_bundle_validation_helpers import (
    operator_contract_audit_state as _operator_contract_audit_state_helper,
)
from pyimgano.reporting.deploy_bundle_validation_helpers import (
    validate_artifact_refs as _validate_artifact_refs_helper,
)
from pyimgano.reporting.deploy_bundle_validation_helpers import (
    validate_exact_mapping as _validate_exact_mapping_helper,
)
from pyimgano.reporting.deploy_bundle_validation_helpers import (
    validate_operator_contract_digests_map as _validate_operator_contract_digests_map_helper,
)
from pyimgano.reporting.deploy_bundle_validation_helpers import (
    validate_required_presence_flag as _validate_required_presence_flag_helper,
)
from pyimgano.reporting.deploy_bundle_validation_helpers import (
    validate_weight_audit_files as _validate_weight_audit_files_helper,
)
from pyimgano.utils.security import FileHasher

DEPLOY_BUNDLE_SCHEMA_VERSION = 1
_DEPLOY_BUNDLE_TYPE = "cpu-offline-qc"
_DEPLOY_BUNDLE_STATUS = "draft"
_REPORT_JSON = "report.json"
_CONFIG_JSON = "config.json"
_ENVIRONMENT_JSON = "environment.json"
_INFER_CONFIG_JSON = "infer_config.json"
_CALIBRATION_CARD_JSON = "calibration_card.json"
_OPERATOR_CONTRACT_JSON = "operator_contract.json"
_MODEL_CARD_JSON = "model_card.json"
_WEIGHTS_MANIFEST_JSON = "weights_manifest.json"
_REQUIRED_SOURCE_RUN_ARTIFACTS = ("report", "config", "environment", "infer_config")
_REQUIRED_BUNDLE_ARTIFACTS = ("report", "config", "environment", "infer_config")

_SOURCE_RUN_ARTIFACT_PATHS = {
    "report": _REPORT_JSON,
    "config": _CONFIG_JSON,
    "environment": _ENVIRONMENT_JSON,
    "infer_config": f"artifacts/{_INFER_CONFIG_JSON}",
    "calibration_card": f"artifacts/{_CALIBRATION_CARD_JSON}",
    "operator_contract": f"artifacts/{_OPERATOR_CONTRACT_JSON}",
}

_BUNDLE_ARTIFACT_PATHS = {
    "report": _REPORT_JSON,
    "config": _CONFIG_JSON,
    "environment": _ENVIRONMENT_JSON,
    "infer_config": _INFER_CONFIG_JSON,
    "calibration_card": _CALIBRATION_CARD_JSON,
    "operator_contract": _OPERATOR_CONTRACT_JSON,
    "model_card": _MODEL_CARD_JSON,
    "weights_manifest": _WEIGHTS_MANIFEST_JSON,
}


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json_mapping_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = _load_json_dict(path)
    except Exception:  # noqa: BLE001 - manifest enrichment is best-effort
        return None
    return dict(payload)


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _extract_split_fingerprint_sha256(payload: Mapping[str, Any]) -> str | None:
    split_fingerprint = payload.get("split_fingerprint", None)
    if not isinstance(split_fingerprint, Mapping):
        return None
    return _nonempty_str(split_fingerprint.get("sha256", None))


def _extract_threshold_scope(payload: Mapping[str, Any]) -> str | None:
    threshold_context = payload.get("threshold_context", None)
    if isinstance(threshold_context, Mapping):
        scope = _nonempty_str(threshold_context.get("scope", None))
        if scope is not None:
            return scope

    artifact_quality = payload.get("artifact_quality", None)
    if isinstance(artifact_quality, Mapping):
        return _nonempty_str(artifact_quality.get("threshold_scope", None))
    return None


def _threshold_items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    image_threshold = payload.get("image_threshold", None)
    if isinstance(image_threshold, Mapping):
        return [image_threshold]

    per_category = payload.get("per_category", None)
    if isinstance(per_category, Mapping):
        return [item for item in per_category.values() if isinstance(item, Mapping)]
    return []


def _has_threshold_payload(payload: Mapping[str, Any]) -> bool:
    return any(
        isinstance(item.get("threshold", None), (int, float)) for item in _threshold_items(payload)
    )


def _has_threshold_provenance(payload: Mapping[str, Any]) -> bool:
    return any(
        isinstance(item.get("provenance", None), Mapping) for item in _threshold_items(payload)
    )


def _supports_pixel_outputs(infer_payload: Mapping[str, Any] | None) -> bool:
    payload = dict(infer_payload or {})
    artifact_quality = payload.get("artifact_quality", None)
    if isinstance(artifact_quality, Mapping):
        return _nonempty_str(artifact_quality.get("threshold_scope", None)) == "pixel"
    return False


def _build_compatibility_payload() -> dict[str, Any]:
    import pyimgano

    return {
        "schema_family": "deploy-bundle",
        "schema_version": int(DEPLOY_BUNDLE_SCHEMA_VERSION),
        "pyimgano_version": str(getattr(pyimgano, "__version__", "")),
        "cpu_only": True,
    }


def _build_input_contract() -> dict[str, Any]:
    return {
        "supported_sources": ["image_dir", "single_image", "input_manifest.jsonl"],
        "record_fields": ["id", "image_path", "category", "meta"],
    }


def _build_threshold_summary(bundle_root: Path) -> dict[str, Any]:
    calibration_card = _load_json_mapping_if_present(bundle_root / _CALIBRATION_CARD_JSON)
    return {
        "scope": (
            _extract_threshold_scope(calibration_card)
            if isinstance(calibration_card, Mapping)
            else None
        ),
        "has_threshold": (
            _has_threshold_payload(calibration_card)
            if isinstance(calibration_card, Mapping)
            else False
        ),
        "has_threshold_provenance": (
            _has_threshold_provenance(calibration_card)
            if isinstance(calibration_card, Mapping)
            else False
        ),
        "has_split_fingerprint": (
            _extract_split_fingerprint_sha256(calibration_card) is not None
            if isinstance(calibration_card, Mapping)
            else False
        ),
        "split_fingerprint_sha256": (
            _extract_split_fingerprint_sha256(calibration_card)
            if isinstance(calibration_card, Mapping)
            else None
        ),
        "calibration_card_ref": (
            _CALIBRATION_CARD_JSON if (bundle_root / _CALIBRATION_CARD_JSON).is_file() else None
        ),
    }


def _build_output_contract(bundle_root: Path) -> dict[str, Any]:
    infer_payload = _load_json_mapping_if_present(bundle_root / _INFER_CONFIG_JSON)
    supports_pixel_outputs = _supports_pixel_outputs(infer_payload)
    return {
        "primary_result_file": "results.jsonl",
        "batch_summary_file": "run_report.json",
        "supports_pixel_outputs": supports_pixel_outputs,
        "optional_artifacts": (
            ["masks/", "overlays/", "defects_regions.jsonl"] if supports_pixel_outputs else []
        ),
    }


def _build_evaluation_summary(bundle_root: Path) -> dict[str, Any]:
    threshold_summary = _build_threshold_summary(bundle_root)
    output_contract = _build_output_contract(bundle_root)
    return {
        "threshold_scope": threshold_summary.get("scope", None),
        "split_fingerprint_sha256": threshold_summary.get("split_fingerprint_sha256", None),
        "supports_image_decision": bool(
            threshold_summary.get("has_threshold")
            and threshold_summary.get("has_threshold_provenance")
        ),
        "supports_pixel_outputs": bool(output_contract.get("supports_pixel_outputs")),
    }


def _build_operator_contract_digests(
    *,
    source_root: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    source_contract = _load_json_mapping_if_present(
        source_root / "artifacts" / _OPERATOR_CONTRACT_JSON
    )
    bundle_contract = _load_json_mapping_if_present(bundle_root / _OPERATOR_CONTRACT_JSON)
    bundle_infer = _load_json_mapping_if_present(bundle_root / _INFER_CONFIG_JSON)
    infer_contract = (
        dict(bundle_infer.get("operator_contract"))
        if isinstance(bundle_infer, Mapping)
        and isinstance(bundle_infer.get("operator_contract"), Mapping)
        else None
    )

    source_sha = (
        _canonical_json_sha256(source_contract) if isinstance(source_contract, Mapping) else None
    )
    bundle_sha = (
        _canonical_json_sha256(bundle_contract) if isinstance(bundle_contract, Mapping) else None
    )
    infer_sha = (
        _canonical_json_sha256(infer_contract) if isinstance(infer_contract, Mapping) else None
    )
    bundle_consistent = (
        bool(
            bundle_sha is not None
            and infer_sha is not None
            and dict(infer_contract) == dict(bundle_contract)
        )
        if (bundle_sha is not None or infer_sha is not None)
        else None
    )
    source_matches_bundle = (
        bool(
            source_sha is not None
            and bundle_sha is not None
            and dict(source_contract) == dict(bundle_contract)
        )
        if (source_sha is not None or bundle_sha is not None)
        else None
    )

    return {
        "source_run_operator_contract_sha256": source_sha,
        "bundle_operator_contract_sha256": bundle_sha,
        "bundle_infer_operator_contract_sha256": infer_sha,
        "bundle_operator_contract_consistent": bundle_consistent,
        "source_run_matches_bundle_operator_contract": source_matches_bundle,
    }


def _classify_entry(rel_path: str) -> str:
    if rel_path == _INFER_CONFIG_JSON:
        return "infer_config"
    if rel_path == _CALIBRATION_CARD_JSON:
        return "calibration_card"
    if rel_path == _OPERATOR_CONTRACT_JSON:
        return "operator_contract"
    if rel_path == _MODEL_CARD_JSON:
        return "model_card"
    if rel_path == _WEIGHTS_MANIFEST_JSON:
        return "weights_manifest"
    if rel_path == _REPORT_JSON:
        return "report"
    if rel_path == _CONFIG_JSON:
        return "config"
    if rel_path == _ENVIRONMENT_JSON:
        return "environment"
    if rel_path.endswith(".pt") or rel_path.endswith(".pth") or rel_path.endswith(".onnx"):
        return "checkpoint"
    return "artifact"


def _collect_existing_artifact_refs(
    root: Path,
    *,
    paths: Mapping[str, str],
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for name, rel_path in paths.items():
        if (root / rel_path).is_file():
            refs[str(name)] = str(rel_path)
    return refs


def _build_artifact_roles(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for entry in entries:
        rel_path = entry.get("path", None)
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        role = entry.get("role", None)
        if not isinstance(role, str) or not role.strip():
            role = _classify_entry(rel_path)
        roles.setdefault(str(role), []).append(str(rel_path))
    return {
        str(role): sorted(paths)
        for role, paths in sorted(roles.items(), key=lambda item: str(item[0]))
    }


def _build_artifact_digests(entries: list[dict[str, Any]]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for entry in entries:
        rel_path = _nonempty_str(entry.get("path", None))
        sha256 = _nonempty_str(entry.get("sha256", None))
        if rel_path is None or sha256 is None:
            continue
        digests[rel_path] = sha256
    return dict(sorted(digests.items(), key=lambda item: item[0]))


def _required_artifacts_present(
    refs: Mapping[str, Any], *, required_names: tuple[str, ...]
) -> bool:
    return all(str(name) in refs for name in required_names)


def _validate_artifact_refs(
    refs: Any,
    *,
    field_name: str,
    root: Path,
    entry_paths: set[str] | None = None,
) -> list[str]:
    return _validate_artifact_refs_helper(
        refs,
        field_name=field_name,
        root=root,
        entry_paths=entry_paths,
    )


def _validate_artifact_roles(
    artifact_roles: Any,
    *,
    field_name: str,
    actual_roles: Mapping[str, list[str]],
) -> list[str]:
    if artifact_roles is None:
        return []
    if not isinstance(artifact_roles, Mapping):
        return [f"{field_name} must be a JSON object/dict."]

    normalized: dict[str, list[str]] = {}
    errors: list[str] = []
    for role, paths in artifact_roles.items():
        if not isinstance(paths, list):
            errors.append(f"{field_name}.{role} must be a list of entry paths.")
            continue
        collected: list[str] = []
        for index, rel_path in enumerate(paths):
            if not isinstance(rel_path, str) or not rel_path.strip():
                errors.append(f"{field_name}.{role}[{index}] must be a non-empty string.")
                continue
            collected.append(str(rel_path))
        normalized[str(role)] = sorted(collected)

    if normalized != {
        str(role): sorted(str(path) for path in paths)
        for role, paths in sorted(actual_roles.items(), key=lambda item: str(item[0]))
    }:
        errors.append(f"{field_name} does not match manifest entries.")
    return errors


def _validate_required_presence_flag(
    value: Any,
    *,
    field_name: str,
    actual: bool,
) -> list[str]:
    return _validate_required_presence_flag_helper(
        value,
        field_name=field_name,
        actual=actual,
    )


_OPERATOR_CONTRACT_DIGEST_KEY_TYPES: dict[str, tuple[type, ...]] = {
    "source_run_operator_contract_sha256": (str, type(None)),
    "bundle_operator_contract_sha256": (str, type(None)),
    "bundle_infer_operator_contract_sha256": (str, type(None)),
    "bundle_operator_contract_consistent": (bool, type(None)),
    "source_run_matches_bundle_operator_contract": (bool, type(None)),
}


def _operator_contract_digest_actual(
    *,
    bundle_root: Path,
    source_run_dir: str | None,
) -> tuple[dict[str, Any], bool]:
    source_root = (
        Path(source_run_dir) if isinstance(source_run_dir, str) and source_run_dir.strip() else None
    )
    source_available = bool(source_root is not None and source_root.exists())
    actual = _build_operator_contract_digests(
        source_root=(source_root if source_root is not None else bundle_root),
        bundle_root=bundle_root,
    )
    return actual, source_available
def _validate_operator_contract_digests(
    value: Any,
    *,
    bundle_root: Path,
    source_run_dir: str | None,
) -> list[str]:
    actual, source_available = _operator_contract_digest_actual(
        bundle_root=bundle_root,
        source_run_dir=source_run_dir,
    )
    return _validate_operator_contract_digests_map_helper(
        value,
        actual=actual,
        source_available=source_available,
        key_types=_OPERATOR_CONTRACT_DIGEST_KEY_TYPES,
    )


def _validate_weight_audit_files(bundle_root: Path, *, check_hashes: bool) -> list[str]:
    return _validate_weight_audit_files_helper(bundle_root, check_hashes=bool(check_hashes))


def _validate_operator_contract_consistency(bundle_root: Path) -> list[str]:
    errors: list[str] = []

    infer_config_path = bundle_root / _INFER_CONFIG_JSON
    if not infer_config_path.is_file():
        return errors

    infer_payload = _load_optional_json_with_errors(
        infer_config_path,
        label=_INFER_CONFIG_JSON,
        errors=errors,
    )
    if infer_payload is None:
        return errors

    operator_contract_path = bundle_root / _OPERATOR_CONTRACT_JSON
    operator_contract_payload = _load_optional_json_with_errors(
        operator_contract_path,
        label=_OPERATOR_CONTRACT_JSON,
        errors=errors,
    )

    has_operator_contract_file = operator_contract_path.is_file()
    has_operator_contract_flag, audit_refs = _operator_contract_audit_state(infer_payload)
    infer_operator_contract = infer_payload.get("operator_contract", None)
    has_infer_operator_contract = isinstance(infer_operator_contract, Mapping)

    _append_operator_contract_presence_errors(
        errors,
        audit_refs=audit_refs,
        has_operator_contract_flag=has_operator_contract_flag,
        has_operator_contract_file=has_operator_contract_file,
        has_infer_operator_contract=has_infer_operator_contract,
    )

    if (
        has_operator_contract_file
        and has_infer_operator_contract
        and isinstance(operator_contract_payload, Mapping)
        and dict(infer_operator_contract) != dict(operator_contract_payload)
    ):
        errors.append(
            f"operator_contract mismatch between {_INFER_CONFIG_JSON} and {_OPERATOR_CONTRACT_JSON}."
        )

    return errors


def _load_optional_json_with_errors(
    path: Path,
    *,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_json_dict(path)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        errors.append(f"{label}: {exc}")
        return None


def _operator_contract_audit_state(
    infer_payload: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    return _operator_contract_audit_state_helper(infer_payload)


def _append_operator_contract_presence_errors(
    errors: list[str],
    *,
    audit_refs: Mapping[str, Any],
    has_operator_contract_flag: bool,
    has_operator_contract_file: bool,
    has_infer_operator_contract: bool,
) -> None:
    _append_operator_contract_presence_errors_helper(
        errors,
        audit_refs=audit_refs,
        has_operator_contract_flag=has_operator_contract_flag,
        has_operator_contract_file=has_operator_contract_file,
        has_infer_operator_contract=has_infer_operator_contract,
        infer_config_json=_INFER_CONFIG_JSON,
        operator_contract_json=_OPERATOR_CONTRACT_JSON,
    )


def build_deploy_bundle_manifest(
    *, bundle_dir: str | Path, source_run_dir: str | Path
) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    source_root = Path(source_run_dir)
    env_path = source_root / "environment.json"
    env = _load_json_dict(env_path) if env_path.exists() else {}
    source_artifact_refs = _collect_existing_artifact_refs(
        source_root,
        paths=_SOURCE_RUN_ARTIFACT_PATHS,
    )
    bundle_artifact_refs = _collect_existing_artifact_refs(
        bundle_root,
        paths=_BUNDLE_ARTIFACT_PATHS,
    )

    entries: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "bundle_manifest.json":
            continue
        rel_path = path.relative_to(bundle_root).as_posix()
        entries.append(
            {
                "path": rel_path,
                "role": _classify_entry(rel_path),
                "size_bytes": int(path.stat().st_size),
                "sha256": FileHasher.compute_hash(str(path), algorithm="sha256"),
            }
        )
    artifact_roles = _build_artifact_roles(entries)
    artifact_digests = _build_artifact_digests(entries)

    return {
        "schema_version": int(DEPLOY_BUNDLE_SCHEMA_VERSION),
        "bundle_type": _DEPLOY_BUNDLE_TYPE,
        "status": _DEPLOY_BUNDLE_STATUS,
        "compatibility": _build_compatibility_payload(),
        "input_contract": _build_input_contract(),
        "output_contract": _build_output_contract(bundle_root),
        "threshold_summary": _build_threshold_summary(bundle_root),
        "evaluation_summary": _build_evaluation_summary(bundle_root),
        "source_run": {
            "run_dir": str(source_root),
            "environment_fingerprint_sha256": env.get("fingerprint_sha256", None),
            "artifact_refs": source_artifact_refs,
        },
        "bundle_artifact_refs": bundle_artifact_refs,
        "artifact_roles": artifact_roles,
        "artifact_digests": artifact_digests,
        "required_source_artifacts_present": _required_artifacts_present(
            source_artifact_refs,
            required_names=_REQUIRED_SOURCE_RUN_ARTIFACTS,
        ),
        "required_bundle_artifacts_present": _required_artifacts_present(
            bundle_artifact_refs,
            required_names=_REQUIRED_BUNDLE_ARTIFACTS,
        ),
        "operator_contract_digests": _build_operator_contract_digests(
            source_root=source_root,
            bundle_root=bundle_root,
        ),
        "entries": entries,
    }


def _validate_exact_mapping(
    value: Any,
    *,
    field_name: str,
    expected: Mapping[str, Any],
) -> list[str]:
    return _validate_exact_mapping_helper(
        value,
        field_name=field_name,
        expected=expected,
    )


def _validate_bundle_type(value: Any) -> list[str]:
    if value is None:
        return []
    if value != _DEPLOY_BUNDLE_TYPE:
        return [f"bundle_type must be {_DEPLOY_BUNDLE_TYPE!r}."]
    return []


def _validate_bundle_status(value: Any) -> list[str]:
    if value is None:
        return []
    if value != _DEPLOY_BUNDLE_STATUS:
        return [f"status must be {_DEPLOY_BUNDLE_STATUS!r}."]
    return []


def _entry_role(rel_path: str, role: Any, *, index: int, errors: list[str]) -> str:
    if role is None:
        return _classify_entry(rel_path)
    if not isinstance(role, str) or not role.strip():
        errors.append(f"entries[{index}].role must be a non-empty string.")
        return _classify_entry(rel_path)
    return str(role)


def _validate_entry_hash(
    entry: Mapping[str, Any],
    *,
    file_path: Path,
    rel_path: str,
    check_hashes: bool,
) -> str | None:
    if not check_hashes:
        return None
    expected = entry.get("sha256", None)
    if not isinstance(expected, str) or not expected.strip():
        return None
    actual = FileHasher.compute_hash(str(file_path), algorithm="sha256")
    if actual != expected:
        return f"SHA256 mismatch for bundled file: {rel_path}"
    return None


def _collect_manifest_entries(
    entries: list[Any],
    *,
    bundle_root: Path,
    check_hashes: bool,
) -> tuple[list[str], set[str], list[dict[str, Any]]]:
    errors: list[str] = []
    entry_paths: set[str] = set()
    actual_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"entries[{index}] must be an object.")
            continue
        rel_path = entry.get("path", None)
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"entries[{index}] missing path.")
            continue
        rel_path_str = str(rel_path)
        entry_paths.add(rel_path_str)
        file_path = bundle_root / rel_path_str
        if not file_path.exists():
            errors.append(f"Missing bundled file: {rel_path_str}")
            continue
        actual_entries.append(
            {
                "path": rel_path_str,
                "role": _entry_role(
                    rel_path_str,
                    entry.get("role", None),
                    index=index,
                    errors=errors,
                ),
            }
        )
        hash_error = _validate_entry_hash(
            entry,
            file_path=file_path,
            rel_path=rel_path_str,
            check_hashes=check_hashes,
        )
        if hash_error is not None:
            errors.append(hash_error)
    return errors, entry_paths, actual_entries


def _source_run_context(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None, list[str]]:
    errors: list[str] = []
    source_artifact_refs: Mapping[str, Any] = {}
    source_run_dir: str | None = None
    source_run = manifest.get("source_run", None)
    if not isinstance(source_run, Mapping):
        return source_artifact_refs, source_run_dir, errors

    source_run_dir_value = source_run.get("run_dir", None)
    if isinstance(source_run_dir_value, str) and source_run_dir_value.strip():
        source_run_dir = str(source_run_dir_value)
    artifact_refs = source_run.get("artifact_refs", None)
    if isinstance(artifact_refs, Mapping):
        source_artifact_refs = artifact_refs
    if isinstance(source_run_dir, str) and source_run_dir:
        errors.extend(
            _validate_artifact_refs(
                artifact_refs,
                field_name="source_run.artifact_refs",
                root=Path(source_run_dir),
            )
        )
    return source_artifact_refs, source_run_dir, errors


def validate_deploy_bundle_manifest(
    manifest: dict[str, Any], *, bundle_dir: str | Path, check_hashes: bool = False
) -> list[str]:
    errors: list[str] = []
    bundle_root = Path(bundle_dir)

    if int(manifest.get("schema_version", 0) or 0) != int(DEPLOY_BUNDLE_SCHEMA_VERSION):
        errors.append("Unsupported deploy bundle manifest schema_version.")

    entries = manifest.get("entries", None)
    if not isinstance(entries, list):
        return ["Deploy bundle manifest entries must be a list."]

    entry_errors, entry_paths, actual_entries = _collect_manifest_entries(
        entries,
        bundle_root=bundle_root,
        check_hashes=bool(check_hashes),
    )
    errors.extend(entry_errors)

    actual_roles = _build_artifact_roles(actual_entries)
    actual_artifact_digests = _build_artifact_digests(entries)
    source_artifact_refs, source_run_dir, source_errors = _source_run_context(manifest)
    errors.extend(source_errors)

    bundle_artifact_refs = manifest.get("bundle_artifact_refs", None)
    errors.extend(_validate_bundle_type(manifest.get("bundle_type", None)))
    errors.extend(_validate_bundle_status(manifest.get("status", None)))
    errors.extend(
        _validate_exact_mapping(
            manifest.get("compatibility", None),
            field_name="compatibility",
            expected=_build_compatibility_payload(),
        )
    )
    errors.extend(
        _validate_exact_mapping(
            manifest.get("input_contract", None),
            field_name="input_contract",
            expected=_build_input_contract(),
        )
    )
    errors.extend(
        _validate_exact_mapping(
            manifest.get("output_contract", None),
            field_name="output_contract",
            expected=_build_output_contract(bundle_root),
        )
    )
    errors.extend(
        _validate_exact_mapping(
            manifest.get("threshold_summary", None),
            field_name="threshold_summary",
            expected=_build_threshold_summary(bundle_root),
        )
    )
    errors.extend(
        _validate_exact_mapping(
            manifest.get("evaluation_summary", None),
            field_name="evaluation_summary",
            expected=_build_evaluation_summary(bundle_root),
        )
    )
    errors.extend(
        _validate_exact_mapping(
            manifest.get("artifact_digests", None),
            field_name="artifact_digests",
            expected=actual_artifact_digests,
        )
    )
    errors.extend(
        _validate_artifact_refs(
            bundle_artifact_refs,
            field_name="bundle_artifact_refs",
            root=bundle_root,
            entry_paths=entry_paths,
        )
    )
    errors.extend(
        _validate_artifact_roles(
            manifest.get("artifact_roles", None),
            field_name="artifact_roles",
            actual_roles=actual_roles,
        )
    )
    errors.extend(
        _validate_required_presence_flag(
            manifest.get("required_source_artifacts_present", None),
            field_name="required_source_artifacts_present",
            actual=_required_artifacts_present(
                source_artifact_refs,
                required_names=_REQUIRED_SOURCE_RUN_ARTIFACTS,
            ),
        )
    )
    errors.extend(
        _validate_required_presence_flag(
            manifest.get("required_bundle_artifacts_present", None),
            field_name="required_bundle_artifacts_present",
            actual=_required_artifacts_present(
                (bundle_artifact_refs if isinstance(bundle_artifact_refs, Mapping) else {}),
                required_names=_REQUIRED_BUNDLE_ARTIFACTS,
            ),
        )
    )
    errors.extend(
        _validate_operator_contract_digests(
            manifest.get("operator_contract_digests", None),
            bundle_root=bundle_root,
            source_run_dir=source_run_dir,
        )
    )
    errors.extend(_validate_operator_contract_consistency(bundle_root))
    errors.extend(_validate_weight_audit_files(bundle_root, check_hashes=bool(check_hashes)))

    return errors


__all__ = [
    "DEPLOY_BUNDLE_SCHEMA_VERSION",
    "build_deploy_bundle_manifest",
    "validate_deploy_bundle_manifest",
]
