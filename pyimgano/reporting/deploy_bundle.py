from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.utils.security import FileHasher
from pyimgano.weights.manifest import validate_weights_manifest_file
from pyimgano.weights.model_card import validate_model_card_file

DEPLOY_BUNDLE_SCHEMA_VERSION = 1
_DEPLOY_BUNDLE_TYPE = "cpu-offline-qc"
_DEPLOY_BUNDLE_STATUS = "draft"
_REQUIRED_SOURCE_RUN_ARTIFACTS = ("report", "config", "environment", "infer_config")
_REQUIRED_BUNDLE_ARTIFACTS = ("report", "config", "environment", "infer_config")

_SOURCE_RUN_ARTIFACT_PATHS = {
    "report": "report.json",
    "config": "config.json",
    "environment": "environment.json",
    "infer_config": "artifacts/infer_config.json",
    "calibration_card": "artifacts/calibration_card.json",
    "operator_contract": "artifacts/operator_contract.json",
}

_BUNDLE_ARTIFACT_PATHS = {
    "report": "report.json",
    "config": "config.json",
    "environment": "environment.json",
    "infer_config": "infer_config.json",
    "calibration_card": "calibration_card.json",
    "operator_contract": "operator_contract.json",
    "model_card": "model_card.json",
    "weights_manifest": "weights_manifest.json",
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
    calibration_card = _load_json_mapping_if_present(bundle_root / "calibration_card.json")
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
            "calibration_card.json" if (bundle_root / "calibration_card.json").is_file() else None
        ),
    }


def _build_output_contract(bundle_root: Path) -> dict[str, Any]:
    infer_payload = _load_json_mapping_if_present(bundle_root / "infer_config.json")
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
        source_root / "artifacts" / "operator_contract.json"
    )
    bundle_contract = _load_json_mapping_if_present(bundle_root / "operator_contract.json")
    bundle_infer = _load_json_mapping_if_present(bundle_root / "infer_config.json")
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
    if rel_path == "infer_config.json":
        return "infer_config"
    if rel_path == "calibration_card.json":
        return "calibration_card"
    if rel_path == "operator_contract.json":
        return "operator_contract"
    if rel_path == "model_card.json":
        return "model_card"
    if rel_path == "weights_manifest.json":
        return "weights_manifest"
    if rel_path == "report.json":
        return "report"
    if rel_path == "config.json":
        return "config"
    if rel_path == "environment.json":
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
    if value is None:
        return []
    if not isinstance(value, bool):
        return [f"{field_name} must be a boolean."]
    if bool(value) != bool(actual):
        return [f"{field_name} does not match manifest artifact refs."]
    return []


def _validate_operator_contract_digests(
    value: Any,
    *,
    bundle_root: Path,
    source_run_dir: str | None,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return ["operator_contract_digests must be a JSON object/dict."]

    source_root = (
        Path(source_run_dir) if isinstance(source_run_dir, str) and source_run_dir.strip() else None
    )
    source_available = bool(source_root is not None and source_root.exists())
    actual = _build_operator_contract_digests(
        source_root=(source_root if source_root is not None else bundle_root),
        bundle_root=bundle_root,
    )
    errors: list[str] = []
    key_types: dict[str, tuple[type, ...]] = {
        "source_run_operator_contract_sha256": (str, type(None)),
        "bundle_operator_contract_sha256": (str, type(None)),
        "bundle_infer_operator_contract_sha256": (str, type(None)),
        "bundle_operator_contract_consistent": (bool, type(None)),
        "source_run_matches_bundle_operator_contract": (bool, type(None)),
    }
    for key, expected_types in key_types.items():
        if key not in value:
            continue
        expected_value = value.get(key, None)
        if not isinstance(expected_value, expected_types):
            errors.append(f"operator_contract_digests.{key} has invalid type.")
            continue
        if (
            key.startswith("source_run_")
            and not bool(source_available)
            and expected_value is not None
        ):
            continue
        if expected_value != actual.get(key, None):
            errors.append(
                f"operator_contract_digests.{key} does not match computed bundle/source contract digest."
            )
    return errors


def _validate_weight_audit_files(bundle_root: Path, *, check_hashes: bool) -> list[str]:
    errors: list[str] = []

    manifest_path = bundle_root / "weights_manifest.json"
    manifest_ok = False
    if manifest_path.is_file():
        try:
            report = validate_weights_manifest_file(
                manifest_path=manifest_path,
                check_files=True,
                check_hashes=bool(check_hashes),
            )
        except Exception as exc:  # noqa: BLE001 - validation boundary
            errors.append(f"weights_manifest.json: {exc}")
        else:
            errors.extend(f"weights_manifest.json: {item}" for item in report.errors)
            manifest_ok = bool(report.ok)

    model_card_path = bundle_root / "model_card.json"
    if model_card_path.is_file():
        try:
            report = validate_model_card_file(
                model_card_path,
                manifest_path=(manifest_path if manifest_ok else None),
                check_files=True,
                check_hashes=bool(check_hashes),
            )
        except Exception as exc:  # noqa: BLE001 - validation boundary
            errors.append(f"model_card.json: {exc}")
        else:
            errors.extend(f"model_card.json: {item}" for item in report.errors)

    return errors


def _validate_operator_contract_consistency(bundle_root: Path) -> list[str]:
    errors: list[str] = []

    infer_config_path = bundle_root / "infer_config.json"
    if not infer_config_path.is_file():
        return errors

    try:
        infer_payload = _load_json_dict(infer_config_path)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        errors.append(f"infer_config.json: {exc}")
        return errors

    operator_contract_path = bundle_root / "operator_contract.json"
    operator_contract_payload: dict[str, Any] | None = None
    has_operator_contract_file = operator_contract_path.is_file()
    if has_operator_contract_file:
        try:
            operator_contract_payload = _load_json_dict(operator_contract_path)
        except Exception as exc:  # noqa: BLE001 - validation boundary
            errors.append(f"operator_contract.json: {exc}")

    artifact_quality = infer_payload.get("artifact_quality", None)
    has_operator_contract_flag = False
    audit_refs: dict[str, Any] = {}
    if isinstance(artifact_quality, Mapping):
        has_operator_contract_flag = bool(artifact_quality.get("has_operator_contract", False))
        audit_refs_raw = artifact_quality.get("audit_refs", None)
        if isinstance(audit_refs_raw, Mapping):
            audit_refs = dict(audit_refs_raw)

    infer_operator_contract = infer_payload.get("operator_contract", None)
    has_infer_operator_contract = isinstance(infer_operator_contract, Mapping)

    if has_operator_contract_flag:
        ref = audit_refs.get("operator_contract", None)
        if not isinstance(ref, str) or not str(ref).strip():
            errors.append(
                "infer_config.json artifact_quality.has_operator_contract=true requires "
                "artifact_quality.audit_refs.operator_contract."
            )
        elif str(ref).strip() != "operator_contract.json":
            errors.append(
                "infer_config.json artifact_quality.audit_refs.operator_contract must point to "
                "operator_contract.json inside deploy bundle."
            )
        if not has_operator_contract_file:
            errors.append(
                "infer_config.json artifact_quality.has_operator_contract=true requires "
                "operator_contract.json in deploy bundle."
            )
        if not has_infer_operator_contract:
            errors.append(
                "infer_config.json artifact_quality.has_operator_contract=true requires "
                "infer_config.operator_contract payload."
            )

    if (
        has_operator_contract_file
        and not has_infer_operator_contract
        and not has_operator_contract_flag
    ):
        errors.append(
            "operator_contract.json exists but infer_config.json is missing operator_contract payload."
        )

    if (
        has_operator_contract_file
        and has_infer_operator_contract
        and isinstance(operator_contract_payload, Mapping)
    ):
        if dict(infer_operator_contract) != dict(operator_contract_payload):
            errors.append(
                "operator_contract mismatch between infer_config.json and operator_contract.json."
            )

    return errors


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
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{field_name} must be a JSON object/dict."]
    if dict(value) != dict(expected):
        return [f"{field_name} does not match computed bundle contract."]
    return []


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

    entry_paths: set[str] = set()
    actual_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entries[{index}] must be an object.")
            continue
        rel_path = entry.get("path", None)
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"entries[{index}] missing path.")
            continue
        entry_paths.add(str(rel_path))
        file_path = bundle_root / rel_path
        if not file_path.exists():
            errors.append(f"Missing bundled file: {rel_path}")
            continue
        role = entry.get("role", None)
        if role is None:
            actual_role = _classify_entry(str(rel_path))
        elif not isinstance(role, str) or not role.strip():
            errors.append(f"entries[{index}].role must be a non-empty string.")
            actual_role = _classify_entry(str(rel_path))
        else:
            actual_role = str(role)
        actual_entries.append({"path": str(rel_path), "role": str(actual_role)})
        if check_hashes:
            expected = entry.get("sha256", None)
            if isinstance(expected, str) and expected.strip():
                actual = FileHasher.compute_hash(str(file_path), algorithm="sha256")
                if actual != expected:
                    errors.append(f"SHA256 mismatch for bundled file: {rel_path}")
    actual_roles = _build_artifact_roles(actual_entries)
    actual_artifact_digests = _build_artifact_digests(entries)

    source_run = manifest.get("source_run", None)
    source_artifact_refs: Mapping[str, Any] = {}
    source_run_dir: str | None = None
    if isinstance(source_run, Mapping):
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
