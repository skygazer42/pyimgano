from __future__ import annotations

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


__all__ = [
    "validate_artifact_refs",
    "validate_exact_mapping",
    "validate_operator_contract_digests_map",
    "validate_required_presence_flag",
    "validate_weight_audit_files",
]
