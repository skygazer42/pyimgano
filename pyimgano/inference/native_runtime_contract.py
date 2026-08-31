from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from pyimgano.inference.artifact_runtime import ArtifactRuntimeError

_CUDA_DEVICE_RE = re.compile(r"cuda(?::([0-9]+))?\Z")
_CANONICAL_PROVIDER_NAMES = frozenset({"CPU", "CUDA"})


def _provider_key(spec: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": str(spec["name"]), "options": dict(spec.get("options", {}))},
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_device_id(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ArtifactRuntimeError(f"{field} must be an integer >= 0.")
    if isinstance(value, int):
        device_id = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None:
        try:
            device_id = int(value)
        except ValueError as exc:
            raise ArtifactRuntimeError(f"{field} must be an integer >= 0.") from exc
    else:
        raise ArtifactRuntimeError(f"{field} must be an integer >= 0.")
    if device_id < 0:
        raise ArtifactRuntimeError(f"{field} must be an integer >= 0.")
    return device_id


def _normalize_provider_specs(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ArtifactRuntimeError(f"{field} must be a list of native provider specs.")

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(raw, Mapping):
            raise ArtifactRuntimeError(f"{item_field} must be a native provider spec object.")
        unknown_fields = sorted((key for key in raw if key not in {"name", "options"}), key=str)
        if unknown_fields:
            raise ArtifactRuntimeError(
                f"{item_field} contains unsupported field(s): {unknown_fields!r}."
            )
        raw_name = raw.get("name")
        if not isinstance(raw_name, str) or raw_name not in _CANONICAL_PROVIDER_NAMES:
            raise ArtifactRuntimeError(
                f"{item_field} contains unsupported native provider {raw_name!r}; "
                "expected canonical 'CPU' or 'CUDA'."
            )
        raw_options = raw.get("options", {})
        if not isinstance(raw_options, Mapping):
            raise ArtifactRuntimeError(f"{item_field}.options must be a mapping.")
        options = dict(raw_options)
        permitted_options = {"device_id"} if raw_name == "CUDA" else set()
        unknown_options = sorted(
            (key for key in options if key not in permitted_options),
            key=str,
        )
        if unknown_options:
            raise ArtifactRuntimeError(
                f"{item_field} provider {raw_name!r} contains unsupported option(s): "
                f"{unknown_options!r}."
            )
        normalized_options: dict[str, Any] = {}
        if "device_id" in options:
            normalized_options["device_id"] = _normalize_device_id(
                options["device_id"],
                field=f"{item_field}.options.device_id",
            )
        spec = {"name": raw_name, "options": normalized_options}
        key = _provider_key(spec)
        if key in seen:
            raise ArtifactRuntimeError(f"{field} contains a duplicate provider spec: {spec!r}.")
        seen.add(key)
        specs.append(spec)
    return specs


def _requested_device_spec(device: str) -> dict[str, Any]:
    if not isinstance(device, str):
        raise ArtifactRuntimeError(f"Unsupported native device override: {device!r}.")
    value = device.strip().lower()
    if value in {"cpu", "cpu:0"}:
        return {"name": "CPU", "options": {}}
    if value == "gpu":
        return {"name": "CUDA", "options": {}}
    match = _CUDA_DEVICE_RE.fullmatch(value)
    if match is None:
        raise ArtifactRuntimeError(f"Unsupported native device override: {device!r}.")
    options = (
        {}
        if match.group(1) is None
        else {"device_id": _normalize_device_id(match.group(1), field="device device_id")}
    )
    return {"name": "CUDA", "options": options}


def _constructor_device(spec: Mapping[str, Any]) -> str:
    if spec["name"] == "CPU":
        return "cpu"
    device_id = dict(spec.get("options", {})).get("device_id")
    return "cuda" if device_id is None else f"cuda:{device_id}"


def resolve_native_device(
    *,
    allowed: Any,
    verified: Any,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one exact allowed-and-verified native device before model creation.

    Native manifests use the canonical provider names ``CPU`` and ``CUDA``.
    ONNX names such as ``CPUExecutionProvider`` are intentionally not aliases;
    artifacts emitted with those names must be migrated at the manifest/export
    boundary rather than weakening runtime policy matching.
    """

    allowed_specs = _normalize_provider_specs(
        allowed,
        field="runtime.allowed_providers",
    )
    verified_specs = _normalize_provider_specs(
        verified,
        field="runtime.verified_providers",
    )
    if not allowed_specs or not verified_specs:
        raise ArtifactRuntimeError(
            "Native artifacts require non-empty allowed_providers and verified_providers."
        )

    allowed_keys = {_provider_key(item) for item in allowed_specs}
    verified_keys = {_provider_key(item) for item in verified_specs}
    intersection = [item for item in allowed_specs if _provider_key(item) in verified_keys]
    if not intersection:
        raise ArtifactRuntimeError(
            "Native artifact has no exact allowed-and-verified device provider."
        )
    if not verified_keys.issubset(allowed_keys):
        raise ArtifactRuntimeError(
            "runtime.verified_providers must be an exact subset of allowed_providers."
        )

    if device is None:
        selected = intersection[0]
    else:
        selected = _requested_device_spec(device)
        key = _provider_key(selected)
        if key not in allowed_keys:
            raise ArtifactRuntimeError(f"Native device {device!r} is not allowed by the artifact.")
        if key not in verified_keys:
            raise ArtifactRuntimeError(
                f"Native device {device!r} is not release-verified by the artifact."
            )

    selected = {"name": selected["name"], "options": dict(selected["options"])}
    return _constructor_device(selected), selected
