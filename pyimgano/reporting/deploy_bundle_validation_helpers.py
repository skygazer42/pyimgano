from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.weights.manifest import validate_weights_manifest_file
from pyimgano.weights.model_card import validate_model_card_file


def validate_exact_mapping(
    value: Any,
    *,
    field_name: str,
    expected: Mapping[str, Any],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{field_name} must be a JSON object/dict."]
    if dict(value) != dict(expected):
        return [f"{field_name} does not match computed bundle contract."]
    return []


def validate_artifact_refs(
    refs: Any,
    *,
    field_name: str,
    root: Path,
    entry_paths: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if refs is None:
        return errors
    if not isinstance(refs, Mapping):
        return [f"{field_name} must be a JSON object/dict."]

    for name, rel_path in refs.items():
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"{field_name}.{name} must be a non-empty string.")
            continue
        file_path = root / rel_path
        if not file_path.is_file():
            errors.append(f"{field_name}.{name} points to missing file: {rel_path}")
            continue
        if entry_paths is not None and rel_path not in entry_paths:
            errors.append(f"{field_name}.{name} is not listed in manifest entries: {rel_path}")
    return errors


def validate_required_presence_flag(
    value: Any,
    *,
    field_name: str,
    actual: bool,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, bool):
        return [f"{field_name} must be a boolean."]
    if bool(value) != bool(actual):
        return [f"{field_name} does not match manifest artifact refs."]
    return []


def validate_operator_contract_digests_map(
    value: Any,
    *,
    actual: Mapping[str, Any],
    source_available: bool,
    key_types: Mapping[str, tuple[type, ...]],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return ["operator_contract_digests must be a JSON object/dict."]

    errors: list[str] = []
    for key, expected_types in key_types.items():
        if key not in value:
            continue
        expected_value = value.get(key, None)
        if not isinstance(expected_value, expected_types):
            errors.append(f"operator_contract_digests.{key} has invalid type.")
            continue
        if key.startswith("source_run_") and not source_available and expected_value is not None:
            continue
        if expected_value != actual.get(key, None):
            errors.append(
                f"operator_contract_digests.{key} does not match computed bundle/source contract digest."
            )
    return errors


def validate_weight_audit_files(bundle_root: Path, *, check_hashes: bool) -> list[str]:
    errors: list[str] = []
    manifest_path = bundle_root / "weights_manifest.json"
    model_card_path = bundle_root / "model_card.json"

    manifest_ok = False
    if manifest_path.is_file():
        try:
            report = validate_weights_manifest_file(
                manifest_path=manifest_path,
                check_files=True,
                check_hashes=bool(check_hashes),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"weights_manifest.json: {exc}")
        else:
            errors.extend(f"weights_manifest.json: {item}" for item in report.errors)
            manifest_ok = bool(report.ok)

    if model_card_path.is_file():
        try:
            report = validate_model_card_file(
                model_card_path,
                manifest_path=(manifest_path if manifest_ok else None),
                check_files=True,
                check_hashes=bool(check_hashes),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"model_card.json: {exc}")
        else:
            errors.extend(f"model_card.json: {item}" for item in report.errors)

    return errors


def operator_contract_audit_state(
    infer_payload: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    artifact_quality = infer_payload.get("artifact_quality", None)
    if not isinstance(artifact_quality, Mapping):
        return False, {}
    audit_refs_raw = artifact_quality.get("audit_refs", None)
    audit_refs = dict(audit_refs_raw) if isinstance(audit_refs_raw, Mapping) else {}
    return bool(artifact_quality.get("has_operator_contract", False)), audit_refs


def append_operator_contract_presence_errors(
    errors: list[str],
    *,
    audit_refs: Mapping[str, Any],
    has_operator_contract_flag: bool,
    has_operator_contract_file: bool,
    has_infer_operator_contract: bool,
    infer_config_json: str,
    operator_contract_json: str,
) -> None:
    if has_operator_contract_flag:
        ref = audit_refs.get("operator_contract", None)
        if not isinstance(ref, str) or not str(ref).strip():
            errors.append(
                f"{infer_config_json} artifact_quality.has_operator_contract=true requires "
                "artifact_quality.audit_refs.operator_contract."
            )
        elif str(ref).strip() != operator_contract_json:
            errors.append(
                f"{infer_config_json} artifact_quality.audit_refs.operator_contract must point to "
                f"{operator_contract_json} inside deploy bundle."
            )
        if not has_operator_contract_file:
            errors.append(
                f"{infer_config_json} artifact_quality.has_operator_contract=true requires "
                f"{operator_contract_json} in deploy bundle."
            )
        if not has_infer_operator_contract:
            errors.append(
                f"{infer_config_json} artifact_quality.has_operator_contract=true requires "
                "infer_config.operator_contract payload."
            )

    if (
        has_operator_contract_file
        and not has_infer_operator_contract
        and not has_operator_contract_flag
    ):
        errors.append(
            f"{operator_contract_json} exists but {infer_config_json} is missing operator_contract payload."
        )


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


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
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: {exc}")
        return None


def source_run_context(
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
            validate_artifact_refs(
                artifact_refs,
                field_name="source_run.artifact_refs",
                root=Path(source_run_dir),
            )
        )
    return source_artifact_refs, source_run_dir, errors


def validate_operator_contract_consistency(
    bundle_root: Path,
    *,
    infer_config_json: str,
    operator_contract_json: str,
) -> list[str]:
    errors: list[str] = []

    infer_config_path = bundle_root / infer_config_json
    if not infer_config_path.is_file():
        return errors

    infer_payload = _load_optional_json_with_errors(
        infer_config_path,
        label=infer_config_json,
        errors=errors,
    )
    if infer_payload is None:
        return errors

    operator_contract_path = bundle_root / operator_contract_json
    operator_contract_payload = _load_optional_json_with_errors(
        operator_contract_path,
        label=operator_contract_json,
        errors=errors,
    )

    has_operator_contract_file = operator_contract_path.is_file()
    has_operator_contract_flag, audit_refs = operator_contract_audit_state(infer_payload)
    infer_operator_contract = infer_payload.get("operator_contract", None)
    has_infer_operator_contract = isinstance(infer_operator_contract, Mapping)

    append_operator_contract_presence_errors(
        errors,
        audit_refs=audit_refs,
        has_operator_contract_flag=has_operator_contract_flag,
        has_operator_contract_file=has_operator_contract_file,
        has_infer_operator_contract=has_infer_operator_contract,
        infer_config_json=infer_config_json,
        operator_contract_json=operator_contract_json,
    )

    if (
        has_operator_contract_file
        and has_infer_operator_contract
        and isinstance(operator_contract_payload, Mapping)
        and dict(infer_operator_contract) != dict(operator_contract_payload)
    ):
        errors.append(
            f"operator_contract mismatch between {infer_config_json} and {operator_contract_json}."
        )

    return errors


__all__ = [
    "append_operator_contract_presence_errors",
    "operator_contract_audit_state",
    "source_run_context",
    "validate_artifact_refs",
    "validate_exact_mapping",
    "validate_operator_contract_consistency",
    "validate_operator_contract_digests_map",
    "validate_required_presence_flag",
    "validate_weight_audit_files",
]
