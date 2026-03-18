from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.utils.security import FileHasher
from pyimgano.weights.manifest import validate_weights_manifest_file
from pyimgano.weights.model_card import validate_model_card_file

DEPLOY_BUNDLE_SCHEMA_VERSION = 1
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


def _required_artifacts_present(refs: Mapping[str, Any], *, required_names: tuple[str, ...]) -> bool:
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


def build_deploy_bundle_manifest(*, bundle_dir: str | Path, source_run_dir: str | Path) -> dict[str, Any]:
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

    return {
        "schema_version": int(DEPLOY_BUNDLE_SCHEMA_VERSION),
        "source_run": {
            "run_dir": str(source_root),
            "environment_fingerprint_sha256": env.get("fingerprint_sha256", None),
            "artifact_refs": source_artifact_refs,
        },
        "bundle_artifact_refs": bundle_artifact_refs,
        "artifact_roles": artifact_roles,
        "required_source_artifacts_present": _required_artifacts_present(
            source_artifact_refs,
            required_names=_REQUIRED_SOURCE_RUN_ARTIFACTS,
        ),
        "required_bundle_artifacts_present": _required_artifacts_present(
            bundle_artifact_refs,
            required_names=_REQUIRED_BUNDLE_ARTIFACTS,
        ),
        "entries": entries,
    }


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

    source_run = manifest.get("source_run", None)
    source_artifact_refs: Mapping[str, Any] = {}
    if isinstance(source_run, Mapping):
        source_run_dir = source_run.get("run_dir", None)
        artifact_refs = source_run.get("artifact_refs", None)
        if isinstance(artifact_refs, Mapping):
            source_artifact_refs = artifact_refs
        if isinstance(source_run_dir, str) and source_run_dir.strip():
            errors.extend(
                _validate_artifact_refs(
                    artifact_refs,
                    field_name="source_run.artifact_refs",
                    root=Path(source_run_dir),
                )
            )

    bundle_artifact_refs = manifest.get("bundle_artifact_refs", None)
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
    errors.extend(_validate_weight_audit_files(bundle_root, check_hashes=bool(check_hashes)))

    return errors


__all__ = [
    "DEPLOY_BUNDLE_SCHEMA_VERSION",
    "build_deploy_bundle_manifest",
    "validate_deploy_bundle_manifest",
]
