from __future__ import annotations

"""Strict schema-v1 manifests for relocatable trained artifacts.

The module deliberately keeps the public representation as a normalized ``dict``.
That gives exporters and runtimes a stable, JSON-shaped contract without exposing
implementation-only dataclasses.  Validation is fail-closed: schema-v1 identities,
layout bindings, providers, components, and verification attachments are checked
before a caller can construct a runtime session.
"""

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
ARTIFACT_SCHEMA_FAMILY = "pyimgano-artifact"
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_POLICY_SCHEMA_FAMILY = "pyimgano-artifact-policy"
ARTIFACT_POLICY_SCHEMA_VERSION = 1

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_POLICY_BYTES = 8 * 1024 * 1024
MAX_COMPONENTS = 4096
MAX_ATTACHMENTS = 1024
MAX_COMPONENT_BYTES = 64 * 1024 * 1024 * 1024

_LAYOUTS = {"native_detector", "single_graph", "composite"}
_BACKENDS = {"pyimgano", "onnxruntime", "torchscript", "openvino"}
_COMPONENT_ROLES = {
    "runtime_model",
    "trained_state",
    "external_data",
    "openvino_weights",
}
_SERIALIZATIONS = {
    "safe-data",
    "torchscript",
    "onnx",
    "openvino-ir",
    "executable-trust-required",
}
_SCORE_TRANSFORMS = {
    "identity",
    "select_index",
    "sigmoid",
    "softmax_select",
    "negate",
}
_VERIFICATION_LEVELS = {"runtime_smoke", "reference_parity", "end_to_end"}
_FORBIDDEN_IMPORT_KEYS = {
    "class",
    "class_name",
    "import",
    "import_path",
    "module",
    "module_path",
    "python_class",
    "python_module",
}
_TOP_LEVEL_KEYS = {
    "schema_family",
    "schema_version",
    "runtime_id",
    "policy_id",
    "artifact_id",
    "layout",
    "model",
    "runtime",
    "input_contract",
    "output_contract",
    "components",
    "policy_ref",
    "compatibility",
    "verification",
    "attachments",
    "provenance",
    "producer",
    "composition",
}
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ArtifactManifestError(ValueError):
    """Raised when an artifact manifest is malformed or fails identity checks."""


def _field_error(path: str, message: str) -> ArtifactManifestError:
    return ArtifactManifestError(f"{path}: {message}")


def _normalize_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonical_number(value: int | float) -> str:
    if isinstance(value, bool):
        raise ArtifactManifestError("Boolean values are not JSON numbers.")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ArtifactManifestError("Canonical JSON rejects NaN and infinity.")
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    text = repr(value).lower()
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        text = f"{mantissa}e{int(exponent)}"
    return text


def _canonical_json_text(value: Any, *, path: str = "$") -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _canonical_number(value)
    if isinstance(value, str):
        return json.dumps(_normalize_string(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return (
            "["
            + ",".join(
                _canonical_json_text(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
            + "]"
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        original_keys: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _field_error(path, "JSON object keys must be strings")
            normalized_key = _normalize_string(key)
            if normalized_key in normalized:
                raise _field_error(
                    path,
                    "object keys collide after Unicode NFC normalization: "
                    f"{original_keys[normalized_key]!r} and {key!r}",
                )
            normalized[normalized_key] = item
            original_keys[normalized_key] = key
        chunks: list[str] = []
        for key in sorted(normalized):
            encoded_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            encoded_value = _canonical_json_text(normalized[key], path=f"{path}.{key}")
            chunks.append(f"{encoded_key}:{encoded_value}")
        return "{" + ",".join(chunks) + "}"
    raise _field_error(path, f"unsupported JSON value type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes using the schema-v1 canonical form."""

    return _canonical_json_text(value).encode("utf-8")


def _canonical_clone(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _sha256_identity(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def compute_policy_id(policy: Mapping[str, Any]) -> str:
    """Compute the policy identity over canonical policy JSON."""

    if not isinstance(policy, Mapping):
        raise ArtifactManifestError("policy: expected a JSON object")
    return _sha256_identity(canonical_json_bytes(policy))


def _runtime_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(dict(manifest))
    for key in (
        "runtime_id",
        "policy_id",
        "artifact_id",
        "policy_ref",
        "verification",
        "attachments",
        "provenance",
        "producer",
    ):
        projection.pop(key, None)
    components = projection.pop("components", [])
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes, bytearray)):
        projection["components"] = sorted(
            (copy.deepcopy(item) for item in components),
            key=lambda item: str(item.get("path", "")) if isinstance(item, Mapping) else "",
        )
    else:
        projection["components"] = copy.deepcopy(components)
    return projection


def compute_runtime_id(manifest: Mapping[str, Any]) -> str:
    """Compute the executable identity, excluding policy and report attachments."""

    if not isinstance(manifest, Mapping):
        raise ArtifactManifestError("manifest: expected a JSON object")
    return _sha256_identity(canonical_json_bytes(_runtime_projection(manifest)))


def compute_artifact_id(
    schema_family: str | Mapping[str, Any],
    schema_version: int | None = None,
    runtime_id: str | None = None,
    policy_id: str | None = None,
) -> str:
    """Compute the composite artifact identity.

    For convenience, callers may pass either the four identity values or a manifest
    mapping as the first argument.
    """

    if isinstance(schema_family, Mapping):
        manifest = schema_family
        family_value = str(manifest.get("schema_family", ""))
        version_value = manifest.get("schema_version")
        runtime_value = manifest.get("runtime_id")
        policy_value = manifest.get("policy_id")
    else:
        family_value = str(schema_family)
        version_value = schema_version
        runtime_value = runtime_id
        policy_value = policy_id
    if not isinstance(version_value, int) or isinstance(version_value, bool):
        raise ArtifactManifestError("schema_version: expected an integer")
    if not isinstance(runtime_value, str) or not _IDENTITY_RE.fullmatch(runtime_value):
        raise ArtifactManifestError("runtime_id: expected sha256:<64 lowercase hex characters>")
    if not isinstance(policy_value, str) or not _IDENTITY_RE.fullmatch(policy_value):
        raise ArtifactManifestError("policy_id: expected sha256:<64 lowercase hex characters>")
    identity_tuple = [family_value, version_value, runtime_value, policy_value]
    return _sha256_identity(canonical_json_bytes(identity_tuple))


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _field_error(path, "expected a JSON object")
    return dict(value)


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _field_error(path, "expected a JSON array")
    return list(value)


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _field_error(path, "expected a non-empty string")
    return _normalize_string(value.strip())


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _field_error(path, "expected an integer")
    if minimum is not None and value < minimum:
        raise _field_error(path, f"must be >= {minimum}")
    return value


def _require_digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _HEX_RE.fullmatch(value):
        raise _field_error(path, "expected 64 lowercase hexadecimal SHA-256 characters")
    return value


def _require_identity(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise _field_error(path, "expected sha256:<64 lowercase hex characters>")
    return value


def _validate_relative_path_text(value: Any, path: str) -> str:
    text = _require_nonempty_string(value, path)
    if "\x00" in text or "\\" in text or "//" in text or text.endswith("/"):
        raise _field_error(path, "must be a normalized relative POSIX path")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise _field_error(path, "absolute and drive-qualified paths are forbidden")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise _field_error(path, "dot, dot-dot, and empty path segments are forbidden")
    return text


def _reject_import_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_IMPORT_KEYS:
                raise _field_error(f"{path}.{key}", "Python import/class paths are forbidden")
            _reject_import_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_import_keys(item, f"{path}[{index}]")


def _validate_safe_scalar_mapping(value: Any, path: str) -> dict[str, Any]:
    mapping = _require_mapping(value, path)
    normalized: dict[str, Any] = {}
    for key, item in mapping.items():
        name = _require_nonempty_string(key, f"{path}.<key>")
        if not _SAFE_NAME_RE.fullmatch(name):
            raise _field_error(f"{path}.{name}", "option name contains unsupported characters")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise _field_error(f"{path}.{name}", "provider/session option must be a scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise _field_error(f"{path}.{name}", "provider/session option must be finite")
        normalized[name] = item
    return normalized


def _validate_session_options(
    value: Any,
    backend: str,
    *,
    path: str = "runtime.session_options",
) -> dict[str, Any]:
    if backend != "onnxruntime":
        raise _field_error(path, "schema v1 session options are supported only by onnxruntime")
    options = _require_mapping(value, path)
    allowed = {
        "enable_cpu_mem_arena",
        "enable_mem_pattern",
        "execution_mode",
        "graph_optimization_level",
        "inter_op_num_threads",
        "intra_op_num_threads",
        "log_severity_level",
        "log_verbosity_level",
        "session_config_entries",
    }
    unknown = set(options) - allowed
    if unknown:
        raise _field_error(path, f"unknown or unsafe keys: {sorted(unknown)}")
    for field in (
        "intra_op_num_threads",
        "inter_op_num_threads",
        "log_severity_level",
        "log_verbosity_level",
    ):
        if field in options:
            options[field] = _require_int(options[field], f"{path}.{field}", minimum=0)
    for field in ("enable_mem_pattern", "enable_cpu_mem_arena"):
        if field in options and not isinstance(options[field], bool):
            raise _field_error(f"{path}.{field}", "expected a boolean")
    if "execution_mode" in options and options["execution_mode"] not in {
        "sequential",
        "parallel",
    }:
        raise _field_error(f"{path}.execution_mode", "must be 'sequential' or 'parallel'")
    if "graph_optimization_level" in options and options["graph_optimization_level"] not in {
        "disable",
        "basic",
        "extended",
        "all",
    }:
        raise _field_error(
            f"{path}.graph_optimization_level",
            "must be disable, basic, extended, or all",
        )
    if "session_config_entries" in options:
        entries = _require_mapping(
            options["session_config_entries"], f"{path}.session_config_entries"
        )
        normalized_entries: dict[str, str] = {}
        for key, item in entries.items():
            name = _require_nonempty_string(key, f"{path}.session_config_entries.<key>")
            if not _SAFE_NAME_RE.fullmatch(name):
                raise _field_error(
                    f"{path}.session_config_entries.{name}", "invalid configuration key"
                )
            if not isinstance(item, str):
                raise _field_error(
                    f"{path}.session_config_entries.{name}", "expected a string value"
                )
            normalized_entries[name] = item
        options["session_config_entries"] = normalized_entries
    return options


def _provider_key(provider: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(provider)


def _validate_providers(runtime: dict[str, Any], *, path: str = "runtime") -> None:
    allowed_path = f"{path}.allowed_providers"
    verified_path = f"{path}.verified_providers"
    allowed_raw = _require_list(runtime.get("allowed_providers"), allowed_path)
    verified_raw = _require_list(runtime.get("verified_providers"), verified_path)
    if not allowed_raw:
        raise _field_error(allowed_path, "must not be empty")
    if not verified_raw:
        raise _field_error(verified_path, "must not be empty")

    def normalize(values: Iterable[Any], field: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[bytes] = set()
        for index, value in enumerate(values):
            item = _require_mapping(value, f"{field}[{index}]")
            unknown = set(item) - {"name", "options"}
            if unknown:
                raise _field_error(f"{field}[{index}]", f"unknown keys: {sorted(unknown)}")
            name = _require_nonempty_string(item.get("name"), f"{field}[{index}].name")
            if not _SAFE_NAME_RE.fullmatch(name):
                raise _field_error(f"{field}[{index}].name", "invalid provider name")
            options = _validate_safe_scalar_mapping(
                item.get("options", {}), f"{field}[{index}].options"
            )
            normalized = {"name": name, "options": options}
            key = _provider_key(normalized)
            if key in seen:
                raise _field_error(field, "contains duplicate provider specifications")
            seen.add(key)
            out.append(normalized)
        return out

    allowed = normalize(allowed_raw, allowed_path)
    verified = normalize(verified_raw, verified_path)
    allowed_keys = {_provider_key(item) for item in allowed}
    if any(_provider_key(item) not in allowed_keys for item in verified):
        raise _field_error(
            verified_path,
            "every verified provider/options spec must also appear in allowed_providers",
        )
    runtime["allowed_providers"] = allowed
    runtime["verified_providers"] = verified


def _validate_attachment(value: Any, path: str) -> dict[str, Any]:
    item = _require_mapping(value, path)
    unknown = set(item) - {"path", "size_bytes", "sha256", "role", "media_type"}
    if unknown:
        raise _field_error(path, f"unknown keys: {sorted(unknown)}")
    item["path"] = _validate_relative_path_text(item.get("path"), f"{path}.path")
    item["size_bytes"] = _require_int(item.get("size_bytes"), f"{path}.size_bytes", minimum=0)
    if item["size_bytes"] > MAX_COMPONENT_BYTES:
        raise _field_error(f"{path}.size_bytes", "exceeds the schema-v1 limit")
    item["sha256"] = _require_digest(item.get("sha256"), f"{path}.sha256")
    if "role" in item:
        item["role"] = _require_nonempty_string(item["role"], f"{path}.role")
    if "media_type" in item:
        item["media_type"] = _require_nonempty_string(item["media_type"], f"{path}.media_type")
    return item


def _validate_components(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _require_list(payload.get("components"), "components")
    if not raw:
        raise _field_error("components", "must contain at least one executable component")
    if len(raw) > MAX_COMPONENTS:
        raise _field_error("components", f"must contain at most {MAX_COMPONENTS} entries")
    components: list[dict[str, Any]] = []
    paths: set[str] = set()
    ids: set[str] = set()
    for index, value in enumerate(raw):
        path = f"components[{index}]"
        item = _require_mapping(value, path)
        unknown = set(item) - {
            "id",
            "path",
            "role",
            "format",
            "serialization",
            "size_bytes",
            "sha256",
        }
        if unknown:
            raise _field_error(path, f"unknown keys: {sorted(unknown)}")
        if "id" in item:
            component_id = _require_nonempty_string(item["id"], f"{path}.id")
            if not _SAFE_NAME_RE.fullmatch(component_id):
                raise _field_error(f"{path}.id", "contains unsupported characters")
            if component_id in ids:
                raise _field_error(f"{path}.id", "duplicate component id")
            ids.add(component_id)
            item["id"] = component_id
        item["path"] = _validate_relative_path_text(item.get("path"), f"{path}.path")
        if item["path"] in paths:
            raise _field_error(f"{path}.path", "duplicate component path")
        paths.add(item["path"])
        item["role"] = _require_nonempty_string(item.get("role"), f"{path}.role")
        if item["role"] not in _COMPONENT_ROLES:
            raise _field_error(f"{path}.role", f"unknown role {item['role']!r}")
        item["format"] = _require_nonempty_string(item.get("format"), f"{path}.format")
        item["serialization"] = _require_nonempty_string(
            item.get("serialization"), f"{path}.serialization"
        )
        if item["serialization"] not in _SERIALIZATIONS:
            raise _field_error(
                f"{path}.serialization", f"unknown serialization {item['serialization']!r}"
            )
        item["size_bytes"] = _require_int(item.get("size_bytes"), f"{path}.size_bytes", minimum=0)
        if item["size_bytes"] > MAX_COMPONENT_BYTES:
            raise _field_error(f"{path}.size_bytes", "exceeds the schema-v1 limit")
        item["sha256"] = _require_digest(item.get("sha256"), f"{path}.sha256")
        components.append(item)
    payload["components"] = components
    return components


def _validate_score_contract(output_contract: dict[str, Any]) -> None:
    score = _require_mapping(output_contract.get("score"), "output_contract.score")
    score["name"] = _require_nonempty_string(score.get("name"), "output_contract.score.name")
    transform = _require_nonempty_string(score.get("transform"), "output_contract.score.transform")
    if transform not in _SCORE_TRANSFORMS:
        raise _field_error("output_contract.score.transform", f"unknown transform {transform!r}")
    score["transform"] = transform
    order = _require_nonempty_string(score.get("score_order"), "output_contract.score.score_order")
    if order not in {"higher_is_more_anomalous", "lower_is_more_anomalous"}:
        raise _field_error(
            "output_contract.score.score_order",
            "must be 'higher_is_more_anomalous' or 'lower_is_more_anomalous'",
        )
    score["score_order"] = order
    if transform in {"select_index", "softmax_select"}:
        _require_int(score.get("axis"), "output_contract.score.axis")
        _require_int(score.get("index"), "output_contract.score.index")
    output_contract["score"] = score


def _validate_model(model: Any, path: str = "model") -> dict[str, Any]:
    value = _require_mapping(model, path)
    unknown = set(value) - {"registry_name", "category", "constructor_kwargs", "asset_bindings"}
    if unknown:
        raise _field_error(path, f"unknown keys: {sorted(unknown)}")
    value["registry_name"] = _require_nonempty_string(
        value.get("registry_name"), f"{path}.registry_name"
    )
    if "category" in value:
        value["category"] = _require_nonempty_string(value["category"], f"{path}.category")
    if "constructor_kwargs" in value:
        value["constructor_kwargs"] = _require_mapping(
            value["constructor_kwargs"], f"{path}.constructor_kwargs"
        )
    if "asset_bindings" in value:
        bindings = _require_mapping(value["asset_bindings"], f"{path}.asset_bindings")
        for key, rel_path in bindings.items():
            binding_name = _require_nonempty_string(key, f"{path}.asset_bindings.<key>")
            segments = binding_name.split(".")
            if any(not segment or not _SAFE_NAME_RE.fullmatch(segment) for segment in segments):
                raise _field_error(
                    f"{path}.asset_bindings.{binding_name}",
                    "binding name must contain safe dot-separated constructor keys",
                )
            forbidden = [
                segment
                for segment in segments
                if segment.strip().lower().replace("-", "_") in _FORBIDDEN_IMPORT_KEYS
            ]
            if forbidden:
                raise _field_error(
                    f"{path}.asset_bindings.{binding_name}",
                    "Python import/class bindings are forbidden",
                )
            _validate_relative_path_text(rel_path, f"{path}.asset_bindings.{key}")
        value["asset_bindings"] = bindings
    _reject_import_keys(value, path)
    return value


def _validate_compatibility(value: Any, layout: str) -> dict[str, Any]:
    compatibility = _require_mapping(value, "compatibility")
    allowed_keys = {
        "pyimgano",
        "python",
        "platforms",
        "runtime_versions",
        "adapter",
        "codecs",
        "onnx_opset",
        "onnx_ir",
    }
    unknown = set(compatibility) - allowed_keys
    if unknown:
        raise _field_error("compatibility", f"unknown keys: {sorted(unknown)}")
    for field in ("pyimgano", "python"):
        compatibility[field] = _require_nonempty_string(
            compatibility.get(field), f"compatibility.{field}"
        )
    platforms = _require_list(compatibility.get("platforms"), "compatibility.platforms")
    if not platforms:
        raise _field_error("compatibility.platforms", "must not be empty")
    from pyimgano.artifacts.compatibility import (
        ArtifactCompatibilityError,
        normalize_platform_tag,
        parse_compatibility_requirements,
    )

    normalized_platforms: list[str] = []
    seen_platforms: set[str] = set()
    for index, item in enumerate(platforms):
        platform = _require_nonempty_string(item, f"compatibility.platforms[{index}]")
        try:
            platform = normalize_platform_tag(platform)
        except ArtifactCompatibilityError as exc:
            raise _field_error(f"compatibility.platforms[{index}]", str(exc)) from exc
        if platform in seen_platforms:
            raise _field_error(
                "compatibility.platforms",
                "contains duplicate normalized platform tags",
            )
        seen_platforms.add(platform)
        normalized_platforms.append(platform)
    compatibility["platforms"] = normalized_platforms
    runtime_versions = _require_mapping(
        compatibility.get("runtime_versions", {}), "compatibility.runtime_versions"
    )
    compatibility["runtime_versions"] = {
        _require_nonempty_string(
            key, "compatibility.runtime_versions.<key>"
        ): _require_nonempty_string(spec, f"compatibility.runtime_versions.{key}")
        for key, spec in runtime_versions.items()
    }
    adapter = compatibility.get("adapter")
    if layout in {"native_detector", "composite"} and adapter is None:
        raise _field_error("compatibility.adapter", f"required for {layout}")
    if adapter is not None:
        adapter_map = _require_mapping(adapter, "compatibility.adapter")
        unknown_adapter = set(adapter_map) - {"id", "version"}
        if unknown_adapter:
            raise _field_error("compatibility.adapter", f"unknown keys: {sorted(unknown_adapter)}")
        adapter_map["id"] = _require_nonempty_string(
            adapter_map.get("id"), "compatibility.adapter.id"
        )
        adapter_map["version"] = _require_int(
            adapter_map.get("version"), "compatibility.adapter.version", minimum=1
        )
        compatibility["adapter"] = adapter_map
    codecs = _require_list(compatibility.get("codecs", []), "compatibility.codecs")
    normalized_codecs: list[dict[str, Any]] = []
    seen_codecs: set[tuple[str, int]] = set()
    for index, codec in enumerate(codecs):
        codec_map = _require_mapping(codec, f"compatibility.codecs[{index}]")
        unknown_codec = set(codec_map) - {"id", "version"}
        if unknown_codec:
            raise _field_error(
                f"compatibility.codecs[{index}]",
                f"unknown keys: {sorted(unknown_codec)}",
            )
        codec_id = _require_nonempty_string(
            codec_map.get("id"), f"compatibility.codecs[{index}].id"
        )
        codec_version = _require_int(
            codec_map.get("version"), f"compatibility.codecs[{index}].version", minimum=1
        )
        key = (codec_id, codec_version)
        if key in seen_codecs:
            raise _field_error("compatibility.codecs", "contains duplicate codec identities")
        seen_codecs.add(key)
        normalized_codecs.append({"id": codec_id, "version": codec_version})
    if layout == "native_detector" and not normalized_codecs:
        raise _field_error("compatibility.codecs", "native_detector requires a safe state codec")
    compatibility["codecs"] = normalized_codecs
    for field in ("onnx_opset", "onnx_ir"):
        if field in compatibility:
            compatibility[field] = _require_int(
                compatibility[field], f"compatibility.{field}", minimum=1
            )
    if ("onnx_opset" in compatibility) != ("onnx_ir" in compatibility):
        raise _field_error(
            "compatibility",
            "onnx_opset and onnx_ir must be declared together",
        )
    try:
        parse_compatibility_requirements(compatibility)
    except ArtifactCompatibilityError as exc:
        raise _field_error("compatibility", str(exc)) from exc
    return compatibility


def _validate_embedding_node(
    node: dict[str, Any],
    *,
    path: str,
    component: Mapping[str, Any],
) -> None:
    runtime_path = f"{path}.runtime"
    child_runtime = _require_mapping(node.get("runtime"), runtime_path)
    unknown_runtime = set(child_runtime) - {
        "backend",
        "allowed_providers",
        "verified_providers",
        "session_options",
    }
    if unknown_runtime:
        raise _field_error(runtime_path, f"unknown keys: {sorted(unknown_runtime)}")
    backend = _require_nonempty_string(child_runtime.get("backend"), f"{runtime_path}.backend")
    if backend not in {"onnxruntime", "torchscript"}:
        raise _field_error(
            f"{runtime_path}.backend",
            "embedding operation requires onnxruntime or torchscript",
        )
    child_runtime["backend"] = backend
    _validate_providers(child_runtime, path=runtime_path)
    if "session_options" in child_runtime:
        child_runtime["session_options"] = _validate_session_options(
            child_runtime["session_options"],
            backend,
            path=f"{runtime_path}.session_options",
        )

    expected_component = {
        "onnxruntime": ("runtime_model", "onnx", "onnx"),
        "torchscript": (
            "runtime_model",
            "torchscript",
            "executable-trust-required",
        ),
    }[backend]
    actual_component = (
        component.get("role"),
        component.get("format"),
        component.get("serialization"),
    )
    if actual_component != expected_component:
        raise _field_error(
            f"{path}.component",
            f"{backend} embedding requires component role/format/serialization "
            f"{expected_component!r}",
        )

    input_contract = _require_mapping(node.get("input_contract"), f"{path}.input_contract")
    if (
        _require_nonempty_string(input_contract.get("kind"), f"{path}.input_contract.kind")
        != "image_batch"
    ):
        raise _field_error(f"{path}.input_contract.kind", "must be 'image_batch'")
    input_contract["name"] = _require_nonempty_string(
        input_contract.get("name"), f"{path}.input_contract.name"
    )
    if input_contract.get("dtype") != "float32":
        raise _field_error(f"{path}.input_contract.dtype", "must be 'float32'")
    if input_contract.get("layout") != "NCHW":
        raise _field_error(f"{path}.input_contract.layout", "must be 'NCHW'")
    if input_contract.get("color_space") != "RGB":
        raise _field_error(f"{path}.input_contract.color_space", "must be 'RGB'")
    size = _require_list(input_contract.get("size"), f"{path}.input_contract.size")
    if len(size) != 2:
        raise _field_error(f"{path}.input_contract.size", "must contain [height, width]")
    input_contract["size"] = [
        _require_int(item, f"{path}.input_contract.size[{index}]", minimum=1)
        for index, item in enumerate(size)
    ]
    node["input_contract"] = input_contract

    output_contract = _require_mapping(node.get("output_contract"), f"{path}.output_contract")
    allowed_output = {"kind", "name", "output_index", "output_key", "reduction"}
    unknown_output = set(output_contract) - allowed_output
    if unknown_output:
        raise _field_error(f"{path}.output_contract", f"unknown keys: {sorted(unknown_output)}")
    if output_contract.get("kind") != "feature_matrix":
        raise _field_error(f"{path}.output_contract.kind", "must be 'feature_matrix'")
    output_contract["name"] = _require_nonempty_string(
        output_contract.get("name"), f"{path}.output_contract.name"
    )
    output_contract["output_index"] = _require_int(
        output_contract.get("output_index", 0),
        f"{path}.output_contract.output_index",
        minimum=0,
    )
    if output_contract.get("reduction") != "auto_2d_v1":
        raise _field_error(f"{path}.output_contract.reduction", "must be 'auto_2d_v1'")
    if "output_key" in output_contract:
        output_contract["output_key"] = _require_nonempty_string(
            output_contract["output_key"], f"{path}.output_contract.output_key"
        )
        if backend != "torchscript":
            raise _field_error(
                f"{path}.output_contract.output_key",
                "is supported only by TorchScript embedding components",
            )
    node["output_contract"] = output_contract
    node["batch_size"] = _require_int(node.get("batch_size"), f"{path}.batch_size", minimum=1)
    node["runtime"] = child_runtime


def _validate_composite(payload: dict[str, Any], components: list[dict[str, Any]]) -> None:
    runtime = payload["runtime"]
    adapter = _require_mapping(runtime.get("composition_adapter"), "runtime.composition_adapter")
    adapter["id"] = _require_nonempty_string(adapter.get("id"), "runtime.composition_adapter.id")
    adapter["version"] = _require_int(
        adapter.get("version"), "runtime.composition_adapter.version", minimum=1
    )
    runtime["composition_adapter"] = adapter
    component_ids = {str(item.get("id")) for item in components if item.get("id") is not None}
    if len(component_ids) != len(components):
        raise _field_error("components", "composite layout requires a unique id on every component")
    components_by_id = {str(item["id"]): item for item in components}

    composition = _require_mapping(payload.get("composition"), "composition")
    unknown_composition = set(composition) - {"nodes", "bindings"}
    if unknown_composition:
        raise _field_error("composition", f"unknown keys: {sorted(unknown_composition)}")
    nodes = _require_list(composition.get("nodes"), "composition.nodes")
    if not nodes:
        raise _field_error("composition.nodes", "must not be empty")
    node_ids: set[str] = set()
    normalized_nodes: list[dict[str, Any]] = []
    explicit_operations = False
    for index, node in enumerate(nodes):
        path = f"composition.nodes[{index}]"
        node_map = _require_mapping(node, path)
        unknown_node = set(node_map) - {
            "id",
            "component",
            "depends_on",
            "operation",
            "runtime",
            "input_contract",
            "output_contract",
            "batch_size",
            "codec",
            "state_model_name",
            "feature_dimension",
        }
        if unknown_node:
            raise _field_error(path, f"unknown keys: {sorted(unknown_node)}")
        node_id = _require_nonempty_string(node_map.get("id"), f"{path}.id")
        if node_id in node_ids:
            raise _field_error(f"{path}.id", "duplicate node id")
        node_ids.add(node_id)
        component_id = _require_nonempty_string(node_map.get("component"), f"{path}.component")
        if component_id not in component_ids:
            raise _field_error(f"{path}.component", "references an unknown component")
        dependencies = _require_list(node_map.get("depends_on", []), f"{path}.depends_on")
        normalized: dict[str, Any] = {
            "id": node_id,
            "component": component_id,
            "depends_on": [
                _require_nonempty_string(dep, f"{path}.depends_on[{dep_index}]")
                for dep_index, dep in enumerate(dependencies)
            ],
        }
        if "operation" in node_map:
            explicit_operations = True
            operation = _require_nonempty_string(node_map["operation"], f"{path}.operation")
            if operation not in {"embedding", "fitted_core"}:
                raise _field_error(f"{path}.operation", f"unknown operation {operation!r}")
            normalized["operation"] = operation
            for key in (
                "runtime",
                "input_contract",
                "output_contract",
                "batch_size",
                "codec",
                "state_model_name",
                "feature_dimension",
            ):
                if key in node_map:
                    normalized[key] = node_map[key]
        elif set(node_map) - {"id", "component", "depends_on"}:
            raise _field_error(path, "operation is required when executable node fields exist")
        normalized_nodes.append(normalized)
    if explicit_operations and any("operation" not in node for node in normalized_nodes):
        raise _field_error(
            "composition.nodes", "explicit executable DAGs require operation on every node"
        )
    for node in normalized_nodes:
        for dependency in node["depends_on"]:
            if dependency not in node_ids:
                raise _field_error(
                    f"composition.nodes.{node['id']}.depends_on",
                    f"references unknown node {dependency!r}",
                )

    dependencies_by_node = {item["id"]: item["depends_on"] for item in normalized_nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise _field_error("composition.nodes", "execution DAG contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in dependencies_by_node[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in dependencies_by_node:
        visit(node_id)
    bindings = _require_mapping(composition.get("bindings"), "composition.bindings")
    for name, node_id in bindings.items():
        _require_nonempty_string(name, "composition.bindings.<key>")
        selected = _require_nonempty_string(node_id, f"composition.bindings.{name}")
        if selected not in node_ids:
            raise _field_error(
                f"composition.bindings.{name}", f"references unknown node {selected!r}"
            )

    if explicit_operations:
        embedding_nodes = [node for node in normalized_nodes if node["operation"] == "embedding"]
        core_nodes = [node for node in normalized_nodes if node["operation"] == "fitted_core"]
        if len(embedding_nodes) != 1 or len(core_nodes) != 1:
            raise _field_error(
                "composition.nodes",
                "schema-v1 executable composites require exactly one embedding and one fitted_core",
            )
        embedding_node, core_node = embedding_nodes[0], core_nodes[0]
        embedding_path = next(
            f"composition.nodes[{index}]"
            for index, item in enumerate(normalized_nodes)
            if item is embedding_node
        )
        core_path = next(
            f"composition.nodes[{index}]"
            for index, item in enumerate(normalized_nodes)
            if item is core_node
        )
        if embedding_node["depends_on"]:
            raise _field_error(f"{embedding_path}.depends_on", "embedding must be the DAG root")
        if core_node["depends_on"] != [embedding_node["id"]]:
            raise _field_error(
                f"{core_path}.depends_on", "fitted_core must depend exactly on embedding"
            )
        embedding_only = {
            "runtime",
            "input_contract",
            "output_contract",
            "batch_size",
        }
        core_only = {"codec", "state_model_name", "feature_dimension"}
        if any(key in embedding_node for key in core_only):
            raise _field_error(embedding_path, "embedding node contains fitted_core fields")
        if any(key in core_node for key in embedding_only):
            raise _field_error(core_path, "fitted_core node contains embedding runtime fields")
        _validate_embedding_node(
            embedding_node,
            path=embedding_path,
            component=components_by_id[embedding_node["component"]],
        )
        core_component = components_by_id[core_node["component"]]
        if (
            core_component.get("role"),
            core_component.get("format"),
            core_component.get("serialization"),
        ) != ("trained_state", "pyimgano-state", "safe-data"):
            raise _field_error(
                f"{core_path}.component",
                "fitted_core requires a safe pyimgano-state trained_state component",
            )
        codec = _require_mapping(core_node.get("codec"), f"{core_path}.codec")
        if set(codec) != {"id", "version"}:
            raise _field_error(f"{core_path}.codec", "requires exactly id and version")
        codec["id"] = _require_nonempty_string(codec.get("id"), f"{core_path}.codec.id")
        codec["version"] = _require_int(
            codec.get("version"), f"{core_path}.codec.version", minimum=1
        )
        core_node["codec"] = codec
        core_node["state_model_name"] = _require_nonempty_string(
            core_node.get("state_model_name"), f"{core_path}.state_model_name"
        )
        core_node["feature_dimension"] = _require_int(
            core_node.get("feature_dimension"), f"{core_path}.feature_dimension", minimum=1
        )
        expected_bindings = {
            "input": embedding_node["id"],
            "features": embedding_node["id"],
            "score": core_node["id"],
        }
        if bindings != expected_bindings:
            raise _field_error(
                "composition.bindings",
                f"executable embedding/core DAG requires exact bindings {expected_bindings!r}",
            )

        model = payload["model"]
        if core_node["state_model_name"] != model["registry_name"]:
            raise _field_error(f"{core_path}.state_model_name", "must equal model.registry_name")
        graph_component = components_by_id[embedding_node["component"]]
        asset_bindings = model.get("asset_bindings", {})
        expected_asset = asset_bindings.get("embedding_kwargs.checkpoint_path")
        if expected_asset != graph_component["path"]:
            raise _field_error(
                "model.asset_bindings.embedding_kwargs.checkpoint_path",
                "must bind the executable embedding component path",
            )
        component_paths = {str(item["path"]) for item in components}
        for binding_name, bound_path in asset_bindings.items():
            if str(bound_path) not in component_paths:
                raise _field_error(
                    f"model.asset_bindings.{binding_name}",
                    "must reference a declared component path",
                )
        if canonical_json_bytes(payload["input_contract"]) != canonical_json_bytes(
            embedding_node["input_contract"]
        ):
            raise _field_error(
                "input_contract", "must equal the executable embedding node input contract"
            )
        if canonical_json_bytes(runtime["allowed_providers"]) != canonical_json_bytes(
            embedding_node["runtime"]["allowed_providers"]
        ) or canonical_json_bytes(runtime["verified_providers"]) != canonical_json_bytes(
            embedding_node["runtime"]["verified_providers"]
        ):
            raise _field_error(
                "runtime.allowed_providers",
                "composite orchestration provider specs must equal the embedding component specs",
            )

    composition["nodes"] = normalized_nodes
    composition["bindings"] = bindings
    payload["composition"] = composition


def _validate_layout(payload: dict[str, Any], components: list[dict[str, Any]]) -> None:
    layout = payload["layout"]
    runtime = payload["runtime"]
    backend = runtime["backend"]
    runtime_models = [item for item in components if item["role"] == "runtime_model"]
    trained_states = [item for item in components if item["role"] == "trained_state"]

    if layout == "native_detector":
        if backend != "pyimgano":
            raise _field_error("layout", "native_detector requires runtime.backend='pyimgano'")
        payload["model"] = _validate_model(payload.get("model"))
        if not trained_states:
            raise _field_error("components", "native_detector requires a trained_state component")
        entrypoint = _validate_relative_path_text(runtime.get("entrypoint"), "runtime.entrypoint")
        if entrypoint not in {item["path"] for item in trained_states}:
            raise _field_error(
                "runtime.entrypoint",
                "native_detector entrypoint must bind a trained_state component",
            )
        runtime["entrypoint"] = entrypoint
        if "composition" in payload or "composition_adapter" in runtime:
            raise _field_error("layout", "native_detector cannot declare composite fields")
        return

    if layout == "single_graph":
        if backend not in {"onnxruntime", "torchscript", "openvino"}:
            raise _field_error(
                "layout", "single_graph requires onnxruntime, torchscript, or openvino backend"
            )
        if len(runtime_models) != 1:
            raise _field_error("components", "single_graph requires exactly one runtime_model")
        entrypoint = _validate_relative_path_text(runtime.get("entrypoint"), "runtime.entrypoint")
        if entrypoint != runtime_models[0]["path"]:
            raise _field_error(
                "runtime.entrypoint", "must reference the unique runtime_model component"
            )
        runtime["entrypoint"] = entrypoint
        if "model" in payload:
            payload["model"] = _validate_model(payload["model"])
        if "composition" in payload or "composition_adapter" in runtime:
            raise _field_error("layout", "single_graph cannot declare composite fields")
        expected = {
            "onnxruntime": ("onnx", "onnx"),
            "torchscript": ("torchscript", "executable-trust-required"),
            "openvino": ("openvino-ir", "openvino-ir"),
        }[backend]
        actual = (runtime_models[0]["format"], runtime_models[0]["serialization"])
        if actual != expected:
            raise _field_error(
                "components", f"{backend} runtime_model requires format/serialization {expected!r}"
            )
        if backend == "openvino":
            weights = [item for item in components if item["role"] == "openvino_weights"]
            if len(weights) != 1:
                raise _field_error(
                    "components", "openvino requires exactly one openvino_weights component"
                )
            model_path = PurePosixPath(runtime_models[0]["path"])
            if model_path.suffix != ".xml":
                raise _field_error(
                    "components", "openvino runtime_model path must use the .xml suffix"
                )
            expected_weights = model_path.with_suffix(".bin").as_posix()
            if weights[0]["path"] != expected_weights:
                raise _field_error(
                    "components",
                    "openvino_weights path must be the .bin sibling of runtime_model",
                )
            if (weights[0]["format"], weights[0]["serialization"]) != (
                "openvino-weights",
                "safe-data",
            ):
                raise _field_error(
                    "components",
                    "openvino_weights requires format='openvino-weights' and "
                    "serialization='safe-data'",
                )
        return

    if backend != "pyimgano":
        raise _field_error("layout", "composite runtime orchestration requires backend='pyimgano'")
    payload["model"] = _validate_model(payload.get("model"))
    if "entrypoint" in runtime:
        raise _field_error(
            "runtime.entrypoint", "composite uses named DAG bindings, not entrypoint"
        )
    if not runtime_models or not trained_states:
        raise _field_error(
            "components", "composite requires runtime_model and trained_state components"
        )
    _validate_composite(payload, components)


def _validate_manifest_structure(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonical_clone(payload)
    if not isinstance(normalized, dict):
        raise ArtifactManifestError("manifest: expected a JSON object")
    unknown = set(normalized) - _TOP_LEVEL_KEYS
    if unknown:
        raise _field_error("manifest", f"unknown keys: {sorted(unknown)}")
    if normalized.get("schema_family") != ARTIFACT_SCHEMA_FAMILY:
        raise _field_error("schema_family", f"expected {ARTIFACT_SCHEMA_FAMILY!r}")
    if normalized.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise _field_error(
            "schema_version", f"unsupported schema version {normalized.get('schema_version')!r}"
        )
    layout = _require_nonempty_string(normalized.get("layout"), "layout")
    if layout not in _LAYOUTS:
        raise _field_error("layout", f"unknown layout {layout!r}")
    normalized["layout"] = layout
    _reject_import_keys(normalized)

    runtime = _require_mapping(normalized.get("runtime"), "runtime")
    allowed_runtime_keys = {
        "backend",
        "allowed_providers",
        "verified_providers",
        "entrypoint",
        "composition_adapter",
        "session_options",
    }
    unknown_runtime = set(runtime) - allowed_runtime_keys
    if unknown_runtime:
        raise _field_error("runtime", f"unknown keys: {sorted(unknown_runtime)}")
    backend = _require_nonempty_string(runtime.get("backend"), "runtime.backend")
    if backend not in _BACKENDS:
        raise _field_error("runtime.backend", f"unknown backend {backend!r}")
    runtime["backend"] = backend
    _validate_providers(runtime)
    if "session_options" in runtime:
        runtime["session_options"] = _validate_session_options(runtime["session_options"], backend)
    normalized["runtime"] = runtime

    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
        validate_artifact_output_contract,
    )

    try:
        normalized["input_contract"] = validate_artifact_input_contract(
            normalized.get("input_contract"),
            layout=layout,
            backend=backend,
        )
        normalized["output_contract"] = validate_artifact_output_contract(
            normalized.get("output_contract")
        )
    except ArtifactIOContractError as exc:
        raise ArtifactManifestError(str(exc)) from exc

    components = _validate_components(normalized)
    _validate_layout(normalized, components)
    normalized["compatibility"] = _validate_compatibility(normalized.get("compatibility"), layout)
    if layout == "composite":
        runtime_adapter = normalized["runtime"]["composition_adapter"]
        compatibility_adapter = normalized["compatibility"]["adapter"]
        if canonical_json_bytes(runtime_adapter) != canonical_json_bytes(compatibility_adapter):
            raise _field_error(
                "compatibility.adapter",
                "must exactly equal runtime.composition_adapter for composite artifacts",
            )
        executable_nodes = [
            node
            for node in normalized["composition"]["nodes"]
            if node.get("operation") == "fitted_core"
        ]
        if executable_nodes:
            required_codec = executable_nodes[0]["codec"]
            if required_codec not in normalized["compatibility"]["codecs"]:
                raise _field_error(
                    "compatibility.codecs",
                    "must contain the fitted_core node codec identity",
                )

    policy_ref = _require_mapping(normalized.get("policy_ref"), "policy_ref")
    unknown_policy_ref = set(policy_ref) - {"path", "policy_id", "sha256", "size_bytes"}
    if unknown_policy_ref:
        raise _field_error("policy_ref", f"unknown keys: {sorted(unknown_policy_ref)}")
    policy_ref["path"] = _validate_relative_path_text(policy_ref.get("path"), "policy_ref.path")
    if "policy_id" in policy_ref:
        policy_ref["policy_id"] = _require_identity(policy_ref["policy_id"], "policy_ref.policy_id")
    if "sha256" in policy_ref:
        policy_ref["sha256"] = _require_digest(policy_ref["sha256"], "policy_ref.sha256")
    if "size_bytes" in policy_ref:
        policy_ref["size_bytes"] = _require_int(
            policy_ref["size_bytes"], "policy_ref.size_bytes", minimum=0
        )
    normalized["policy_ref"] = policy_ref

    verification = _require_mapping(normalized.get("verification"), "verification")
    unknown_verification = set(verification) - {"level", "reference_backend", "report"}
    if unknown_verification:
        raise _field_error("verification", f"unknown keys: {sorted(unknown_verification)}")
    level = _require_nonempty_string(verification.get("level"), "verification.level")
    if level not in _VERIFICATION_LEVELS:
        raise _field_error(
            "verification.level",
            f"must be one of {sorted(_VERIFICATION_LEVELS)}; structural alone is not deployable",
        )
    verification["level"] = level
    if level in {"reference_parity", "end_to_end"}:
        verification["reference_backend"] = _require_nonempty_string(
            verification.get("reference_backend"), "verification.reference_backend"
        )
    elif "reference_backend" in verification:
        raise _field_error(
            "verification.reference_backend", "runtime_smoke must not claim a reference backend"
        )
    verification["report"] = _validate_attachment(verification.get("report"), "verification.report")
    normalized["verification"] = verification

    attachments = normalized.get("attachments", [])
    if not isinstance(attachments, list):
        raise _field_error("attachments", "expected a JSON array")
    if len(attachments) > MAX_ATTACHMENTS:
        raise _field_error("attachments", f"must contain at most {MAX_ATTACHMENTS} entries")
    normalized["attachments"] = [
        _validate_attachment(item, f"attachments[{index}]")
        for index, item in enumerate(attachments)
    ]
    provenance_attachments: list[dict[str, Any]] = []
    for optional_mapping in ("provenance", "producer"):
        if optional_mapping not in normalized:
            continue
        metadata = _require_mapping(normalized[optional_mapping], optional_mapping)
        if "attachments" in metadata:
            raw_provenance_attachments = _require_list(
                metadata["attachments"], f"{optional_mapping}.attachments"
            )
            if len(raw_provenance_attachments) > MAX_ATTACHMENTS:
                raise _field_error(
                    f"{optional_mapping}.attachments",
                    f"must contain at most {MAX_ATTACHMENTS} entries",
                )
            validated = [
                _validate_attachment(item, f"{optional_mapping}.attachments[{index}]")
                for index, item in enumerate(raw_provenance_attachments)
            ]
            metadata["attachments"] = validated
            provenance_attachments.extend(validated)
        normalized[optional_mapping] = metadata

    seen_paths: dict[str, str] = {}

    def register_path(path: str, owner: str) -> None:
        previous = seen_paths.get(path)
        if previous is not None:
            raise _field_error(
                owner, f"duplicate referenced path {path!r}; already owned by {previous}"
            )
        seen_paths[path] = owner

    for index, component in enumerate(components):
        register_path(component["path"], f"components[{index}].path")
    register_path(policy_ref["path"], "policy_ref.path")
    register_path(verification["report"]["path"], "verification.report.path")
    for index, attachment in enumerate(normalized["attachments"]):
        register_path(attachment["path"], f"attachments[{index}].path")
    for index, attachment in enumerate(provenance_attachments):
        register_path(attachment["path"], f"provenance.attachments[{index}].path")
    return normalized


def validate_artifact_manifest(
    payload: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate schema, layout bindings, and all available identity layers.

    Passing ``policy`` is required to independently recompute ``policy_id``.  When
    it is omitted, the declared policy identity remains usable for writer pipelines,
    but ``policy_ref.policy_id`` and the composite identity must still agree.
    ``load_artifact_manifest`` always supplies the artifact-local policy.
    """

    normalized = _validate_manifest_structure(payload)
    declared_runtime = _require_identity(normalized.get("runtime_id"), "runtime_id")
    expected_runtime = compute_runtime_id(normalized)
    if declared_runtime != expected_runtime:
        raise _field_error(
            "runtime_id", f"does not match canonical runtime identity {expected_runtime}"
        )

    declared_policy = _require_identity(normalized.get("policy_id"), "policy_id")
    if policy is not None:
        from pyimgano.artifacts.policy import validate_artifact_policy

        normalized_policy = validate_artifact_policy(policy, manifest_model=normalized.get("model"))
        expected_policy = compute_policy_id(normalized_policy)
        if declared_policy != expected_policy:
            raise _field_error(
                "policy_id", f"does not match canonical policy identity {expected_policy}"
            )
    policy_ref_id = normalized["policy_ref"].get("policy_id")
    if policy_ref_id != declared_policy:
        raise _field_error("policy_ref.policy_id", "must equal manifest policy_id")

    declared_artifact = _require_identity(normalized.get("artifact_id"), "artifact_id")
    expected_artifact = compute_artifact_id(
        ARTIFACT_SCHEMA_FAMILY,
        ARTIFACT_SCHEMA_VERSION,
        declared_runtime,
        declared_policy,
    )
    if declared_artifact != expected_artifact:
        raise _field_error("artifact_id", f"does not match composite identity {expected_artifact}")
    return normalized


def build_artifact_manifest(
    payload: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize a manifest draft and compute runtime, policy, and artifact IDs."""

    if not isinstance(payload, Mapping):
        raise ArtifactManifestError("manifest: expected a JSON object")
    from pyimgano.artifacts.policy import validate_artifact_policy

    draft = copy.deepcopy(dict(payload))
    for identity_key in ("runtime_id", "policy_id", "artifact_id"):
        draft.pop(identity_key, None)
    normalized_policy = validate_artifact_policy(policy, manifest_model=draft.get("model"))
    policy_bytes = canonical_json_bytes(normalized_policy)
    policy_id = compute_policy_id(normalized_policy)
    policy_ref = _require_mapping(draft.get("policy_ref"), "policy_ref")
    policy_ref["policy_id"] = policy_id
    policy_ref["sha256"] = hashlib.sha256(policy_bytes).hexdigest()
    policy_ref["size_bytes"] = len(policy_bytes)
    draft["policy_ref"] = policy_ref

    # Structural validation precedes identity construction.  Temporary identities
    # satisfy the validator's shape requirements but are never returned.
    placeholder = f"sha256:{'0' * 64}"
    draft["runtime_id"] = placeholder
    draft["policy_id"] = policy_id
    draft["artifact_id"] = placeholder
    normalized = _validate_manifest_structure(draft)
    runtime_id = compute_runtime_id(normalized)
    normalized["runtime_id"] = runtime_id
    normalized["policy_id"] = policy_id
    normalized["policy_ref"]["policy_id"] = policy_id
    normalized["artifact_id"] = compute_artifact_id(
        ARTIFACT_SCHEMA_FAMILY,
        ARTIFACT_SCHEMA_VERSION,
        runtime_id,
        policy_id,
    )
    return validate_artifact_manifest(normalized, normalized_policy)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    normalized_keys: set[str] = set()
    for key, value in pairs:
        normalized = _normalize_string(key)
        if normalized in normalized_keys:
            raise ArtifactManifestError(f"duplicate JSON object key: {key!r}")
        normalized_keys.add(normalized)
        result[normalized] = value
    return result


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_json_no_duplicates)
    except ArtifactManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"{label}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactManifestError(f"{label}: expected a JSON object")
    return value


def _manifest_path(path: str | Path) -> Path:
    value = Path(path)
    if (
        value.is_dir()
        or value.name != ARTIFACT_MANIFEST_FILENAME
        and value.suffix.lower() != ".json"
    ):
        return value / ARTIFACT_MANIFEST_FILENAME
    return value


def _read_bounded_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    if path.is_symlink():
        raise ArtifactManifestError(f"{label}: symlinks are forbidden: {path}")
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    if not path.is_file():
        raise ArtifactManifestError(f"{label}: expected a regular file: {path}")
    if stat_result.st_size > maximum:
        raise ArtifactManifestError(f"{label}: exceeds {maximum} bytes")
    return path.read_bytes()


def load_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest and its contained policy, verifying all three identities."""

    manifest_path = _manifest_path(path)
    manifest_bytes = _read_bounded_regular_file(
        manifest_path, maximum=MAX_MANIFEST_BYTES, label="artifact manifest"
    )
    payload = _load_json_bytes(manifest_bytes, label="artifact manifest")
    policy_ref = _require_mapping(payload.get("policy_ref"), "policy_ref")
    policy_rel = _validate_relative_path_text(policy_ref.get("path"), "policy_ref.path")

    from pyimgano.artifacts.security import resolve_contained_path

    root = manifest_path.parent.resolve()
    policy_path = resolve_contained_path(root, policy_rel)
    policy_bytes = _read_bounded_regular_file(
        policy_path, maximum=MAX_POLICY_BYTES, label="artifact policy"
    )
    expected_size = policy_ref.get("size_bytes")
    if expected_size is not None and len(policy_bytes) != expected_size:
        raise _field_error(
            "policy_ref.size_bytes",
            f"expected {expected_size}, found {len(policy_bytes)}",
        )
    expected_digest = _require_digest(policy_ref.get("sha256"), "policy_ref.sha256")
    actual_digest = hashlib.sha256(policy_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise _field_error(
            "policy_ref.sha256", f"expected {expected_digest}, found {actual_digest}"
        )
    policy = _load_json_bytes(policy_bytes, label="artifact policy")
    return validate_artifact_manifest(payload, policy)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_artifact_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a canonical manifest, and optionally its canonical policy.

    When ``policy`` is supplied, identities and ``policy_ref`` are rebuilt and the
    policy is written before the manifest.  This preserves the manifest-last commit
    rule used by artifact exporters.
    """

    manifest_path = _manifest_path(path)
    root = manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    if policy is not None:
        manifest = build_artifact_manifest(payload, policy)
        from pyimgano.artifacts.policy import validate_artifact_policy
        from pyimgano.artifacts.security import resolve_contained_path

        normalized_policy = validate_artifact_policy(policy, manifest_model=manifest.get("model"))
        policy_bytes = canonical_json_bytes(normalized_policy)
        policy_path = resolve_contained_path(root, manifest["policy_ref"]["path"], must_exist=False)
        _atomic_write(policy_path, policy_bytes)
    else:
        manifest = validate_artifact_manifest(payload)
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    return manifest_path


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_POLICY_SCHEMA_FAMILY",
    "ARTIFACT_POLICY_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA_FAMILY",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactManifestError",
    "MAX_ATTACHMENTS",
    "MAX_COMPONENTS",
    "MAX_COMPONENT_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_POLICY_BYTES",
    "build_artifact_manifest",
    "canonical_json_bytes",
    "compute_artifact_id",
    "compute_policy_id",
    "compute_runtime_id",
    "load_artifact_manifest",
    "validate_artifact_manifest",
    "write_artifact_manifest",
]
