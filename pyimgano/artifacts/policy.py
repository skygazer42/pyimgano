from __future__ import annotations

"""Artifact-local inference policy authority and transactional rebinding."""

import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from pyimgano.artifacts.manifest import (
    ARTIFACT_POLICY_SCHEMA_FAMILY,
    ARTIFACT_POLICY_SCHEMA_VERSION,
    MAX_POLICY_BYTES,
    ArtifactManifestError,
    build_artifact_manifest,
    canonical_json_bytes,
    load_artifact_manifest,
    write_artifact_manifest,
)
from pyimgano.artifacts.security import (
    ArtifactSecurityError,
    resolve_contained_path,
    stage_verified_artifact,
    verify_artifact_files,
)

_FORBIDDEN_POLICY_KEYS = {
    "artifact_manifest",
    "backend",
    "checkpoint",
    "checkpoint_path",
    "class",
    "class_name",
    "components",
    "entrypoint",
    "from_run",
    "import",
    "import_path",
    "module",
    "module_path",
    "run_dir",
    "runtime",
    "source_run",
    "state_dict",
}
_MODEL_KEYS = {"registry_name", "name", "category", "constructor_kwargs", "asset_bindings"}
_UNSET = object()


class ArtifactPolicyError(ValueError):
    """Raised when policy data exceeds its authority or rebinding is unsafe."""


def _error(path: str, message: str) -> ArtifactPolicyError:
    return ArtifactPolicyError(f"{path}: {message}")


def _reject_forbidden_keys(value: Any, path: str = "policy") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_POLICY_KEYS:
                raise _error(
                    f"{path}.{key}",
                    "artifact policy cannot own executable reconstruction or external run state",
                )
            _reject_forbidden_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}[{index}]")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "expected a JSON object")
    return dict(value)


def _finite_optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(path, "expected a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "expected a finite number or null")
    return result


def _normalized_model(value: Any, path: str) -> dict[str, Any]:
    model = _mapping(value, path)
    unknown = set(model) - _MODEL_KEYS
    if unknown:
        raise _error(path, f"unknown model mirror keys: {sorted(unknown)}")
    registry_name = model.get("registry_name")
    legacy_name = model.get("name")
    if registry_name is not None and legacy_name is not None and registry_name != legacy_name:
        raise _error(path, "name and registry_name mirrors conflict")
    selected = registry_name if registry_name is not None else legacy_name
    if not isinstance(selected, str) or not selected.strip():
        raise _error(f"{path}.registry_name", "expected a non-empty registry name")
    normalized: dict[str, Any] = {"registry_name": selected.strip()}
    if model.get("category") is not None:
        if not isinstance(model["category"], str) or not model["category"].strip():
            raise _error(f"{path}.category", "expected a non-empty category")
        normalized["category"] = model["category"].strip()
    if "constructor_kwargs" in model:
        normalized["constructor_kwargs"] = _mapping(
            model["constructor_kwargs"], f"{path}.constructor_kwargs"
        )
    if "asset_bindings" in model:
        bindings = _mapping(model["asset_bindings"], f"{path}.asset_bindings")
        for key, rel_path in bindings.items():
            if not isinstance(key, str) or not key:
                raise _error(f"{path}.asset_bindings", "binding names must be non-empty strings")
            # Use a temporary empty root only for lexical validation would be
            # misleading; the manifest security layer validates bound assets.
            if not isinstance(rel_path, str) or not rel_path or ".." in rel_path.split("/"):
                raise _error(f"{path}.asset_bindings.{key}", "must be a contained relative path")
        normalized["asset_bindings"] = bindings
    return normalized


def _normalized_manifest_model(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _normalized_model(value, "manifest.model")


def _load_policy_input(policy: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(policy, Mapping):
        return policy
    if not isinstance(policy, (str, Path)):
        raise ArtifactPolicyError("policy: expected a JSON object or JSON file path")
    path = Path(policy)
    if path.is_symlink():
        raise ArtifactPolicyError(f"Policy JSON path must not be a symlink: {path}")
    try:
        info = path.stat()
    except FileNotFoundError:
        raise ArtifactPolicyError(f"Policy JSON file not found: {path}") from None
    if not path.is_file():
        raise ArtifactPolicyError(f"Policy JSON path is not a regular file: {path}")
    if info.st_size > MAX_POLICY_BYTES:
        raise ArtifactPolicyError(f"Policy JSON exceeds {MAX_POLICY_BYTES} bytes: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactPolicyError(f"Policy JSON contains duplicate key: {key!r}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        if len(raw) > MAX_POLICY_BYTES:
            raise ArtifactPolicyError(f"Policy JSON exceeds {MAX_POLICY_BYTES} bytes: {path}")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except ArtifactPolicyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactPolicyError(f"Policy JSON is invalid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactPolicyError(f"Policy JSON root must be an object: {path}")
    return payload


def validate_artifact_policy(
    payload: Mapping[str, Any],
    *,
    manifest_model: Mapping[str, Any] | None | object = _UNSET,
) -> dict[str, Any]:
    """Validate and normalize policy-owned fields and compatibility mirrors."""

    if not isinstance(payload, Mapping):
        raise ArtifactPolicyError("policy: expected a JSON object")
    try:
        normalized = json.loads(canonical_json_bytes(payload).decode("utf-8"))
    except ArtifactManifestError as exc:
        raise ArtifactPolicyError(str(exc)) from exc
    if not isinstance(normalized, dict):
        raise ArtifactPolicyError("policy: expected a JSON object")
    if normalized.get("schema_family") != ARTIFACT_POLICY_SCHEMA_FAMILY:
        raise _error("policy.schema_family", f"expected {ARTIFACT_POLICY_SCHEMA_FAMILY!r}")
    if normalized.get("schema_version") != ARTIFACT_POLICY_SCHEMA_VERSION:
        raise _error(
            "policy.schema_version",
            f"unsupported version {normalized.get('schema_version')!r}",
        )
    _reject_forbidden_keys(normalized)

    expected_model = (
        None if manifest_model is _UNSET else _normalized_manifest_model(manifest_model)
    )
    policy_model: dict[str, Any] | None = None
    if "model" in normalized:
        policy_model = _normalized_model(normalized["model"], "policy.model")
        normalized["model"] = policy_model
    if manifest_model is None and policy_model is not None:
        raise _error("policy.model", "third-party single-graph artifacts must omit model mirror")
    if expected_model is not None:
        if policy_model is not None and canonical_json_bytes(policy_model) != canonical_json_bytes(
            expected_model
        ):
            raise _error("policy.model", "does not match authoritative manifest.model")

    postprocess = _mapping(normalized.get("postprocess"), "policy.postprocess")
    image_threshold = _mapping(
        postprocess.get("image_threshold"), "policy.postprocess.image_threshold"
    )
    canonical_threshold = _finite_optional_number(
        image_threshold.get("threshold"), "policy.postprocess.image_threshold.threshold"
    )
    image_threshold["threshold"] = canonical_threshold
    if "score_order" in image_threshold:
        if image_threshold["score_order"] != "higher_is_more_anomalous":
            raise _error(
                "policy.postprocess.image_threshold.score_order",
                "must be 'higher_is_more_anomalous'",
            )
    else:
        image_threshold["score_order"] = "higher_is_more_anomalous"
    postprocess["image_threshold"] = image_threshold

    if "threshold" in normalized:
        mirror_threshold = _finite_optional_number(normalized["threshold"], "policy.threshold")
        if mirror_threshold != canonical_threshold:
            raise _error(
                "policy.threshold",
                "legacy threshold mirror conflicts with postprocess.image_threshold.threshold",
            )
        normalized["threshold"] = mirror_threshold
    if "pixel_threshold" in normalized:
        canonical_pixel = postprocess.get("pixel_threshold")
        if canonical_json_bytes(normalized["pixel_threshold"]) != canonical_json_bytes(
            canonical_pixel
        ):
            raise _error(
                "policy.pixel_threshold",
                "legacy pixel_threshold mirror conflicts with postprocess.pixel_threshold",
            )
    normalized["postprocess"] = postprocess
    return normalized


def write_artifact_policy(
    path: str | Path,
    policy: Mapping[str, Any],
    *,
    manifest_model: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a canonical artifact policy."""

    normalized = validate_artifact_policy(policy, manifest_model=manifest_model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(normalized)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _default_probe(path: Path, *, trust_checkpoint: bool = False) -> Any:
    try:
        module = importlib.import_module("pyimgano.inference.artifact_runtime")
        callback = getattr(module, "probe_artifact_policy")
    except (ImportError, AttributeError) as exc:
        raise ArtifactPolicyError(
            "Mandatory runtime/policy probe is unavailable; pass the runtime probe callback."
        ) from exc
    return callback(path, trust_checkpoint=bool(trust_checkpoint))


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, directories, files in os.walk(root):
        current = Path(directory)
        current.chmod(0o700)
        for name in directories:
            (current / name).chmod(0o700)
        for name in files:
            (current / name).chmod(0o600)


def bind_policy(
    source: str | Path,
    policy: Mapping[str, Any] | str | Path,
    out: str | Path,
    *,
    probe: Callable[[Path], Any] | None = None,
    trust_checkpoint: bool = False,
) -> Path:
    """Clone an artifact and bind a new policy as one atomic publication.

    The executable files are cloned from verified private staging, so the source is
    never trusted via a verify-then-reopen sequence.  ``runtime_id`` must remain
    unchanged.  The supplied runtime probe runs before the destination is published.
    """

    source_path = Path(source)
    source_manifest_path = (
        source_path / "artifact_manifest.json" if source_path.is_dir() else source_path
    )
    source_root = source_manifest_path.parent
    destination = Path(out)
    if destination.exists():
        raise ArtifactPolicyError(f"Destination already exists: {destination}")
    try:
        source_manifest = load_artifact_manifest(source_manifest_path)
        policy_payload = _load_policy_input(policy)
        normalized_policy = validate_artifact_policy(
            policy_payload, manifest_model=source_manifest.get("model")
        )
        verify_artifact_files(source_root, source_manifest)
    except (ArtifactManifestError, ArtifactSecurityError, FileNotFoundError) as exc:
        raise ArtifactPolicyError(f"Cannot bind policy from source artifact: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.bind-", dir=destination.parent))
    try:
        with stage_verified_artifact(source_root, source_manifest) as staging:
            shutil.copytree(
                staging.root, temporary, dirs_exist_ok=True, copy_function=shutil.copyfile
            )
        _make_writable(temporary)

        draft = copy.deepcopy(source_manifest)
        for key in ("runtime_id", "policy_id", "artifact_id"):
            draft.pop(key, None)
        preliminary = build_artifact_manifest(draft, normalized_policy)
        if preliminary["runtime_id"] != source_manifest["runtime_id"]:
            raise ArtifactPolicyError("Policy rebinding changed runtime_id")
        write_artifact_manifest(temporary, preliminary, policy=normalized_policy)

        callback = probe
        try:
            probe_result = (
                callback(temporary)
                if callback is not None
                else _default_probe(
                    temporary,
                    trust_checkpoint=bool(trust_checkpoint),
                )
            )
        except Exception as exc:  # noqa: BLE001 - mandatory acceptance boundary
            raise ArtifactPolicyError(f"Mandatory runtime/policy probe failed: {exc}") from exc

        verification = copy.deepcopy(preliminary["verification"])
        report_ref = verification["report"]
        report_path = resolve_contained_path(temporary, report_ref["path"], must_exist=False)
        if isinstance(probe_result, Mapping):
            report_payload = dict(probe_result)
        else:
            report_payload = {
                "kind": "runtime_policy_probe",
                "status": "passed",
                "runtime_id": preliminary["runtime_id"],
                "policy_id": preliminary["policy_id"],
                "artifact_id": preliminary["artifact_id"],
            }
        report_bytes = canonical_json_bytes(report_payload)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(report_bytes)
        report_ref["size_bytes"] = len(report_bytes)
        report_ref["sha256"] = hashlib.sha256(report_bytes).hexdigest()
        verification["report"] = report_ref
        draft["verification"] = verification

        final_manifest = build_artifact_manifest(draft, normalized_policy)
        if final_manifest["runtime_id"] != source_manifest["runtime_id"]:
            raise ArtifactPolicyError("Policy rebinding changed runtime_id")
        write_artifact_manifest(temporary, final_manifest, policy=normalized_policy)
        load_artifact_manifest(temporary)
        verify_artifact_files(temporary, final_manifest)

        if destination.exists():
            raise ArtifactPolicyError(f"Destination already exists: {destination}")
        os.rename(temporary, destination)
        return destination
    except ArtifactPolicyError:
        raise
    except (ArtifactManifestError, ArtifactSecurityError, FileNotFoundError, OSError) as exc:
        raise ArtifactPolicyError(f"Policy binding failed: {exc}") from exc
    finally:
        if temporary.exists():
            _make_writable(temporary)
            shutil.rmtree(temporary)


__all__ = [
    "ArtifactPolicyError",
    "bind_policy",
    "validate_artifact_policy",
    "write_artifact_policy",
]
