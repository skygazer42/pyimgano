from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyimgano.inference.artifact_runtime import ArtifactRuntime, ArtifactRuntimeError


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _selector_format(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return {"openvino-ir": "openvino"}.get(normalized, normalized)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactRuntimeError(f"Failed to read artifact JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactRuntimeError(f"Artifact JSON must contain an object: {path}")
    return payload


def _safe_child(root: Path, relative: str) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute() or not raw.parts or any(part in {"", ".", ".."} for part in raw.parts):
        raise ArtifactRuntimeError(
            f"Bundle artifact reference is not a safe relative path: {relative!r}"
        )
    candidate = (root / raw).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactRuntimeError(
            f"Bundle artifact reference escapes its root: {relative!r}"
        ) from exc
    return candidate


def _collect_index_candidates(
    value: Any, *, inherited: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    inherited_values = dict(inherited or {})
    candidates: list[dict[str, Any]] = []
    if isinstance(value, str):
        if value.endswith("artifact_manifest.json") or value:
            candidate = dict(inherited_values)
            candidate["path"] = value
            candidates.append(candidate)
        return candidates
    if isinstance(value, list):
        for item in value:
            candidates.extend(_collect_index_candidates(item, inherited=inherited_values))
        return candidates
    if not isinstance(value, Mapping):
        return candidates

    current = dict(inherited_values)
    for key in ("category", "format", "backend", "artifact_id"):
        if value.get(key) is not None:
            current[key] = str(value[key])
    path = value.get("path", value.get("artifact", value.get("artifact_path")))
    if isinstance(path, str):
        candidate = dict(current)
        candidate["path"] = path
        candidates.append(candidate)
    for key, nested in value.items():
        if key in {
            "path",
            "artifact",
            "artifact_path",
            "artifact_id",
            "backend",
            "category",
            "format",
            "index_id",
            "manifest",
            "schema_family",
            "schema_version",
            "sha256",
            "digest",
            "slug",
        }:
            continue
        child_context = dict(current)
        if key not in {"artifacts", "entries", "artifact_refs", "categories", "formats"}:
            if "category" not in child_context:
                child_context["category"] = str(key)
            elif "format" not in child_context:
                child_context["format"] = str(key)
        candidates.extend(_collect_index_candidates(nested, inherited=child_context))
    return candidates


def _candidate_manifest_path(bundle_root: Path, candidate: Mapping[str, Any]) -> Path:
    path = _safe_child(bundle_root, str(candidate["path"]))
    if path.is_dir():
        path = path / "artifact_manifest.json"
    return path


def _resolve_artifact_source(
    source: str | Path,
    *,
    category: str | None,
    artifact_format: str | None,
    backend: str | None,
    artifact_id: str | None,
) -> Path:
    artifact_format = _selector_format(artifact_format)
    path = Path(source)
    if path.is_file() and path.name == "artifact_manifest.json":
        return path
    if path.is_dir() and (path / "artifact_manifest.json").is_file():
        return path / "artifact_manifest.json"
    if path.is_file():
        raise ArtifactRuntimeError(
            f"load_artifact() requires artifact_manifest.json, not {path.name!r}. "
            "Raw ONNX files must first be imported with an explicit contract."
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact path not found: {path}")

    index_paths = [path / "export_index.json", path / "bundle_manifest.json"]
    candidates: list[dict[str, Any]] = []
    for index_path in index_paths:
        if index_path.is_file():
            if index_path.name == "export_index.json":
                try:
                    from pyimgano.artifacts import load_export_index

                    payload = load_export_index(index_path, root=path)
                except Exception as exc:  # noqa: BLE001 - index trust boundary
                    raise ArtifactRuntimeError(f"Invalid export index {index_path}: {exc}") from exc
            else:
                payload = _load_json_object(index_path)
            root_value = payload.get("artifact_refs", payload)
            candidates.extend(_collect_index_candidates(root_value))

    # Keep only actual artifact manifests and de-duplicate by resolved path.
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        try:
            manifest_path = _candidate_manifest_path(path, candidate)
        except ArtifactRuntimeError:
            raise
        if manifest_path.is_file():
            item = dict(candidate)
            item["manifest_path"] = manifest_path
            try:
                manifest_payload = _load_json_object(manifest_path)
            except ArtifactRuntimeError:
                continue
            model = _mapping(manifest_payload.get("model"))
            runtime = _mapping(manifest_payload.get("runtime"))
            actual_category = model.get("category")
            actual_backend = runtime.get("backend")
            runtime_component = _component_by_role(manifest_payload, "runtime_model")
            if runtime_component is not None:
                actual_format = _selector_format(runtime_component.get("format"))
            elif str(manifest_payload.get("layout")) == "native_detector":
                actual_format = "native"
            else:
                actual_format = None
            actual_values = {
                "category": actual_category,
                "backend": actual_backend,
                "format": actual_format,
                "artifact_id": manifest_payload.get("artifact_id"),
            }
            for field, actual in actual_values.items():
                declared = item.get(field)
                declared_for_compare = _selector_format(declared) if field == "format" else declared
                if declared is not None and str(declared_for_compare) != str(actual):
                    raise ArtifactRuntimeError(
                        f"Artifact index {field} conflicts with manifest: "
                        f"index={declared!r}, manifest={actual!r}, path={manifest_path}."
                    )
                if actual is not None:
                    item[field] = actual
            unique[str(manifest_path.resolve())] = item
    filtered = list(unique.values())
    selectors = {
        "category": category,
        "format": artifact_format,
        "backend": backend,
        "artifact_id": artifact_id,
    }
    for key, expected in selectors.items():
        if expected is not None:
            filtered = [item for item in filtered if str(item.get(key)) == str(expected)]
    if not filtered:
        raise ArtifactRuntimeError(
            "No artifact in the bundle matches the requested selectors: "
            f"category={category!r}, format={artifact_format!r}, backend={backend!r}, "
            f"artifact_id={artifact_id!r}."
        )
    if len(filtered) != 1:
        choices = [
            {
                "category": item.get("category"),
                "format": item.get("format"),
                "backend": item.get("backend"),
                "path": str(item.get("path")),
            }
            for item in filtered
        ]
        raise ArtifactRuntimeError(
            f"Artifact selection is ambiguous; provide category/format/backend. Choices: {choices}"
        )
    return Path(filtered[0]["manifest_path"])


def _component_by_role(manifest: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    matches = [
        dict(item)
        for item in manifest.get("components", [])
        if isinstance(item, Mapping) and str(item.get("role")) == role
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ArtifactRuntimeError(f"Artifact declares multiple {role!r} components.")
    return matches[0]


def _component_lookup(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("components", []):
        if not isinstance(item, Mapping):
            continue
        component = dict(item)
        for key in (component.get("id"), component.get("path")):
            if isinstance(key, str) and key:
                result[key] = component
    return result


def _validate_registered_adapter(manifest: Mapping[str, Any]) -> Any | None:
    compatibility = _mapping(manifest.get("compatibility"))
    adapter_spec = compatibility.get("adapter")
    if adapter_spec is None:
        return None
    if not isinstance(adapter_spec, Mapping):
        raise ArtifactRuntimeError("compatibility.adapter must be a mapping.")
    adapter_id = str(adapter_spec.get("id", "")).strip()
    adapter_version = int(adapter_spec.get("version", 0) or 0)

    # Importing the public package performs the explicit, idempotent built-in
    # registration before the private registry is queried.
    import pyimgano.exporting  # noqa: F401
    from pyimgano.exporting.registry import EXPORT_ADAPTER_REGISTRY

    try:
        adapter = EXPORT_ADAPTER_REGISTRY.get_by_id(adapter_id)
    except KeyError as exc:
        raise ArtifactRuntimeError(
            f"Artifact export adapter is not registered in this build: {adapter_id!r}."
        ) from exc
    registered_version = int(getattr(adapter, "adapter_version", 0) or 0)
    if registered_version != adapter_version:
        raise ArtifactRuntimeError(
            "Artifact adapter version does not match the registered implementation: "
            f"manifest={adapter_version}, registered={registered_version}."
        )

    model = _mapping(manifest.get("model"))
    model_name = str(model.get("registry_name", model.get("name", ""))).strip()
    if model_name:
        try:
            model_adapter = EXPORT_ADAPTER_REGISTRY.get(model_name)
        except KeyError as exc:
            raise ArtifactRuntimeError(
                f"Artifact model has no registered export adapter: {model_name!r}."
            ) from exc
        model_identity = (
            str(getattr(model_adapter, "adapter_id", "")),
            int(getattr(model_adapter, "adapter_version", 0) or 0),
        )
        if model_identity != (adapter_id, adapter_version):
            raise ArtifactRuntimeError(
                "Artifact model is bound to a different registered export adapter."
            )

    if str(manifest.get("layout", "")).strip().lower() == "native_detector":
        from packaging.specifiers import InvalidSpecifier, SpecifierSet

        declared = _mapping(compatibility.get("runtime_versions"))
        registered = getattr(adapter, "native_runtime_versions", {})
        if not isinstance(registered, Mapping):
            raise ArtifactRuntimeError(
                "Registered native adapter has an invalid runtime-version contract."
            )
        try:
            declared_specs = {str(key): SpecifierSet(str(value)) for key, value in declared.items()}
            registered_specs = {
                str(key): SpecifierSet(str(value)) for key, value in registered.items()
            }
        except InvalidSpecifier as exc:
            raise ArtifactRuntimeError(
                f"Registered native adapter has an invalid runtime-version specifier: {exc}"
            ) from exc
        if declared_specs != registered_specs:
            raise ArtifactRuntimeError(
                "Artifact native runtime requirements do not exactly match its registered adapter."
            )
    return adapter


def _validate_safe_state_bindings(
    manifest: Mapping[str, Any], *, staged: Any, adapter: Any | None
) -> None:
    compatibility = _mapping(manifest.get("compatibility"))
    declared_raw = compatibility.get("codecs", [])
    if not isinstance(declared_raw, list):
        raise ArtifactRuntimeError("compatibility.codecs must be a list.")
    declared = {
        (str(item.get("id", "")), int(item.get("version", 0) or 0))
        for item in declared_raw
        if isinstance(item, Mapping)
    }
    state_components = [
        dict(item)
        for item in manifest.get("components", [])
        if isinstance(item, Mapping)
        and item.get("role") == "trained_state"
        and item.get("format") == "pyimgano-state"
        and item.get("serialization") == "safe-data"
    ]
    if not state_components:
        if declared:
            raise ArtifactRuntimeError(
                "Artifact declares fitted-state codecs but has no matching safe state component."
            )
        return
    if len(state_components) != 1:
        raise ArtifactRuntimeError("Artifact must contain exactly one safe fitted-state component.")

    from pyimgano.exporting.state_codec import get_state_codec, inspect_fitted_state
    from pyimgano.exporting.types import CheckpointCompleteness

    state = state_components[0]
    try:
        info = inspect_fitted_state(staged.path_for(str(state["path"])))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArtifactRuntimeError(f"Fitted-state identity inspection failed: {exc}") from exc
    actual = {(info.codec_id, int(info.codec_version))}
    if actual != declared:
        raise ArtifactRuntimeError(
            "Fitted-state codec identity does not exactly match compatibility.codecs."
        )
    if info.completeness is not CheckpointCompleteness.COMPLETE:
        raise ArtifactRuntimeError("Fitted-state archive is not certified complete.")
    try:
        codec = get_state_codec(info.codec_id, info.codec_version)
    except KeyError as exc:
        raise ArtifactRuntimeError(
            f"Fitted-state codec is not registered: {info.codec_id!r} v{info.codec_version}."
        ) from exc
    if int(getattr(codec, "state_schema_version", 0) or 0) != int(info.state_schema_version):
        raise ArtifactRuntimeError(
            "Fitted-state schema version does not match the registered codec."
        )
    if info.model_name not in {str(name) for name in getattr(codec, "model_names", ())}:
        raise ArtifactRuntimeError("Fitted-state codec/model binding is invalid.")

    model = _mapping(manifest.get("model"))
    model_name = str(model.get("registry_name", model.get("name", ""))).strip()
    if model_name and info.model_name != model_name:
        raise ArtifactRuntimeError(
            "Fitted-state model identity does not match the artifact model declaration."
        )
    if adapter is not None and str(manifest.get("layout", "")).strip().lower() in {
        "native_detector",
        "composite",
    }:
        expected = (
            str(getattr(adapter, "state_codec_id", "")),
            int(getattr(adapter, "state_codec_version", 0) or 0),
        )
        if next(iter(actual)) != expected:
            raise ArtifactRuntimeError(
                "Fitted-state codec identity does not match the registered adapter."
            )


def _model_constructor_kwargs(
    manifest: Mapping[str, Any],
    *,
    staged: Any,
    device: str | None,
    providers: Any,
    session_options: Any,
) -> tuple[str, dict[str, Any]]:
    model = _mapping(manifest.get("model"))
    name = str(model.get("registry_name", model.get("name", ""))).strip()
    if not name:
        raise ArtifactRuntimeError("Native/composite artifact requires model.registry_name.")
    constructor = _mapping(model.get("constructor"))
    kwargs = _mapping(constructor.get("kwargs"))
    if not kwargs:
        kwargs = _mapping(model.get("constructor_kwargs")) or _mapping(model.get("model_kwargs"))
    components = _component_lookup(manifest)
    bindings = _mapping(model.get("asset_bindings"))
    bindings.update(_mapping(constructor.get("asset_bindings")))
    for argument, reference in bindings.items():
        ref = str(reference)
        component = components.get(ref)
        relative = str(component.get("path")) if component is not None else ref
        kwargs[str(argument)] = str(staged.path_for(relative))
    if device is not None:
        kwargs["device"] = str(device)
    if providers is not None:
        from pyimgano.inference.onnx_runtime import _normalize_provider_specs

        kwargs["providers"] = [
            item["name"] for item in _normalize_provider_specs(providers, field="providers")
        ]
    if session_options is not None:
        kwargs["session_options"] = dict(session_options)
    return name, kwargs


def _load_native_or_composite(
    manifest: Mapping[str, Any],
    *,
    staged: Any,
    device: str | None,
    providers: Any,
    session_options: Mapping[str, Any] | None,
    trust_checkpoint: bool,
) -> tuple[Any, str]:
    if providers is not None:
        raise ArtifactRuntimeError("providers is not supported by native artifacts; use device.")
    if session_options is not None:
        raise ArtifactRuntimeError("session_options is not supported by native artifacts.")
    runtime = _mapping(manifest.get("runtime"))
    from pyimgano.inference.native_runtime_contract import resolve_native_device

    selected_device, selected_provider = resolve_native_device(
        allowed=runtime.get("allowed_providers"),
        verified=runtime.get("verified_providers"),
        device=device,
    )
    name, kwargs = _model_constructor_kwargs(
        manifest,
        staged=staged,
        device=selected_device,
        providers=None,
        session_options=None,
    )
    import pyimgano.models  # noqa: F401 - populate lazy registry
    from pyimgano.models.registry import create_model

    detector = create_model(name, **kwargs)
    state = _component_by_role(manifest, "trained_state")
    if state is not None:
        serialization = str(state.get("serialization", "safe-data"))
        state_path = staged.path_for(str(state["path"]))
        state_format = str(state.get("format", "")).strip().lower()
        if serialization == "safe-data":
            if state_format != "pyimgano-state":
                raise ArtifactRuntimeError(
                    "Unsupported safe trained-state format: "
                    f"{state.get('format')!r}. Expected 'pyimgano-state'."
                )
            from pyimgano.exporting.state_codec import load_fitted_state

            load_fitted_state(detector, state_path, expected_model_name=name)
        elif serialization == "executable-trust-required":
            if not trust_checkpoint:
                raise ArtifactRuntimeError(
                    "Artifact state requires executable deserialization. Reload only with "
                    "trust_checkpoint=True after verifying provenance."
                )
            from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

            load_checkpoint_into_detector(detector, state_path, trusted=True)
        else:
            raise ArtifactRuntimeError(
                f"Unsupported trained-state serialization: {serialization!r}."
            )
    runtime_info = _mapping(getattr(detector, "runtime_info", None))
    runtime_info.update(
        {
            "backend": "pyimgano",
            "device": selected_device,
            "providers": [str(selected_provider["name"])],
            "selected_provider": str(selected_provider["name"]),
        }
    )
    try:
        detector.runtime_info = runtime_info
    except (AttributeError, TypeError):
        pass
    return detector, name


def _component_by_id(manifest: Mapping[str, Any], component_id: str) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in manifest.get("components", [])
        if isinstance(item, Mapping) and str(item.get("id", "")) == str(component_id)
    ]
    if len(matches) != 1:
        raise ArtifactRuntimeError(
            f"Composite DAG component {component_id!r} must resolve exactly once."
        )
    return matches[0]


def _validated_torchscript_component_device(
    child_runtime: Mapping[str, Any], device: str | None
) -> str:
    def specs(field: str) -> list[tuple[str, tuple[tuple[str, Any], ...]]]:
        values = child_runtime.get(field)
        if not isinstance(values, list):
            raise ArtifactRuntimeError(f"Composite child runtime {field} must be a list.")
        result: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
        for raw in values:
            if not isinstance(raw, Mapping):
                raise ArtifactRuntimeError(
                    f"Composite child runtime {field} contains an invalid provider spec."
                )
            options = raw.get("options", {})
            if not isinstance(options, Mapping):
                raise ArtifactRuntimeError(
                    f"Composite child runtime {field} provider options must be a mapping."
                )
            result.append((str(raw.get("name", "")), tuple(sorted(dict(options).items()))))
        return result

    allowed = specs("allowed_providers")
    verified = specs("verified_providers")
    verified_intersection = [item for item in verified if item in set(allowed)]
    if not verified_intersection:
        raise ArtifactRuntimeError(
            "TorchScript composite component has no allowed and release-verified device."
        )
    if device is None:
        provider_name, options = verified_intersection[0]
        selected = {"CPU": "cpu", "CUDA": "cuda"}.get(provider_name)
        if selected is None or options:
            raise ArtifactRuntimeError(
                "TorchScript composite provider cannot be represented as a safe device."
            )
        return selected

    selected = str(device).strip().lower()
    if selected in {"cpu", "cpu:0"}:
        selected, provider_name = "cpu", "CPU"
    elif selected in {"cuda", "cuda:0", "gpu"}:
        selected, provider_name = "cuda", "CUDA"
    else:
        raise ArtifactRuntimeError(
            f"Unsupported TorchScript composite device override: {device!r}."
        )
    selected_key = (provider_name, ())
    if selected_key not in allowed:
        raise ArtifactRuntimeError(
            f"TorchScript composite device {selected!r} is not allowed by its component."
        )
    if selected_key not in verified:
        raise ArtifactRuntimeError(
            f"TorchScript composite device {selected!r} is not release-verified."
        )
    return selected


def _load_composite(
    manifest: Mapping[str, Any],
    *,
    staged: Any,
    device: str | None,
    providers: Any,
    session_options: Mapping[str, Any] | None,
    trust_checkpoint: bool,
) -> tuple[Any, str]:
    composition = _mapping(manifest.get("composition"))
    nodes = composition.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ArtifactRuntimeError("Composite artifact has no executable composition DAG.")
    executable = [dict(item) for item in nodes if isinstance(item, Mapping)]
    embedding_nodes = [item for item in executable if item.get("operation") == "embedding"]
    core_nodes = [item for item in executable if item.get("operation") == "fitted_core"]
    if len(embedding_nodes) != 1 or len(core_nodes) != 1 or len(executable) != 2:
        raise ArtifactRuntimeError(
            "Composite runtime requires an explicit one-embedding/one-core DAG."
        )
    embedding_node, core_node = embedding_nodes[0], core_nodes[0]
    if core_node.get("depends_on") != [embedding_node.get("id")]:
        raise ArtifactRuntimeError("Composite fitted core is not bound to its embedding node.")

    runtime = _mapping(manifest.get("runtime"))
    adapter_spec = _mapping(runtime.get("composition_adapter"))
    adapter_id = str(adapter_spec.get("id", "")).strip()
    adapter_version = int(adapter_spec.get("version", 0) or 0)
    try:
        from pyimgano.exporting.registry import EXPORT_ADAPTER_REGISTRY

        adapter = EXPORT_ADAPTER_REGISTRY.get_by_id(adapter_id)
    except KeyError as exc:
        raise ArtifactRuntimeError(
            f"Composite adapter is not registered in this build: {adapter_id!r}."
        ) from exc
    if int(getattr(adapter, "adapter_version", 0)) != adapter_version:
        raise ArtifactRuntimeError(
            "Composite adapter version does not match the registered implementation."
        )
    compose = getattr(adapter, "compose", None)
    load_core = getattr(adapter, "load_composite_core", None)
    if not callable(compose) or not callable(load_core):
        raise ArtifactRuntimeError(
            f"Composite adapter {adapter_id!r} is missing its executable hooks."
        )

    graph_component = _component_by_id(manifest, str(embedding_node.get("component", "")))
    graph_path = staged.path_for(str(graph_component["path"]))
    child_runtime = _mapping(embedding_node.get("runtime"))
    child_backend = str(child_runtime.get("backend", "")).strip().lower()
    input_contract = _mapping(embedding_node.get("input_contract"))
    output_contract = _mapping(embedding_node.get("output_contract"))
    batch_size = int(embedding_node.get("batch_size", 0) or 0)
    if child_backend == "onnxruntime":
        from pyimgano.artifacts import validate_onnx_graph_contract

        try:
            graph_info = validate_onnx_graph_contract(
                graph_path,
                input_contract=input_contract,
                output_contract=output_contract,
            )
            compatibility = _mapping(manifest.get("compatibility"))
            expected_ir = compatibility.get("onnx_ir")
            expected_opset = compatibility.get("onnx_opset")
            if expected_ir is not None and int(expected_ir) != graph_info.ir_version:
                raise ValueError(
                    "compatibility.onnx_ir does not match the composite embedding graph."
                )
            if expected_opset is not None and int(expected_opset) != graph_info.default_opset:
                raise ValueError(
                    "compatibility.onnx_opset does not match the composite embedding graph."
                )
        except (OSError, ValueError) as exc:
            raise ArtifactRuntimeError(
                f"Composite ONNX graph contract validation failed: {exc}"
            ) from exc
        from pyimgano.inference.composite_runtime import OnnxEmbeddingComponentRuntime

        declared_options = _mapping(child_runtime.get("session_options"))
        selected_options = declared_options
        if session_options is not None:
            if not isinstance(session_options, Mapping):
                raise ArtifactRuntimeError(
                    "Composite ONNX session_options override must be a mapping."
                )
            from pyimgano.artifacts import canonical_json_bytes

            candidate_options = dict(session_options)
            try:
                matches_declared = canonical_json_bytes(candidate_options) == canonical_json_bytes(
                    declared_options
                )
            except Exception as exc:  # noqa: BLE001 - caller configuration boundary
                raise ArtifactRuntimeError(
                    f"Invalid composite ONNX session_options override: {exc}"
                ) from exc
            if not matches_declared:
                raise ArtifactRuntimeError(
                    "Composite ONNX session_options override is not the exact "
                    "release-verified manifest configuration."
                )
        component_runtime = OnnxEmbeddingComponentRuntime(
            graph_path,
            input_contract=input_contract,
            output_contract=output_contract,
            batch_size=batch_size,
            allowed_providers=child_runtime.get("allowed_providers"),
            verified_providers=child_runtime.get("verified_providers"),
            providers=providers,
            device=device,
            session_options=selected_options,
        )
    elif child_backend == "torchscript":
        if not trust_checkpoint:
            raise ArtifactRuntimeError(
                "TorchScript composite execution requires executable deserialization. "
                "Reload with trust_checkpoint=True only after verifying provenance."
            )
        if providers is not None:
            raise ArtifactRuntimeError(
                "providers is not supported by a TorchScript composite component."
            )
        if session_options is not None:
            raise ArtifactRuntimeError(
                "session_options is not supported by a TorchScript composite component."
            )
        selected_device = _validated_torchscript_component_device(child_runtime, device)
        from pyimgano.inference.composite_runtime import (
            TorchScriptEmbeddingComponentRuntime,
        )

        component_runtime = TorchScriptEmbeddingComponentRuntime(
            graph_path,
            input_contract=input_contract,
            output_contract=output_contract,
            batch_size=batch_size,
            device=selected_device,
            trust_checkpoint=True,
        )
    else:
        raise ArtifactRuntimeError(f"Unsupported composite embedding backend: {child_backend!r}.")

    state_component = _component_by_id(manifest, str(core_node.get("component", "")))
    if (
        state_component.get("role"),
        state_component.get("format"),
        state_component.get("serialization"),
    ) != ("trained_state", "pyimgano-state", "safe-data"):
        raise ArtifactRuntimeError("Composite fitted core is not safe pyimgano state.")
    codec = _mapping(core_node.get("codec"))
    model_name = str(core_node.get("state_model_name", "")).strip()
    fitted_core = load_core(
        staged.path_for(str(state_component["path"])),
        model_name=model_name,
        codec_id=str(codec.get("id", "")),
        codec_version=int(codec.get("version", 0) or 0),
    )
    empirical = getattr(fitted_core, "_x_sorted", None)
    feature_dimension = int(core_node.get("feature_dimension", 0) or 0)
    if (
        empirical is None
        or getattr(empirical, "ndim", 0) != 2
        or int(empirical.shape[1]) != feature_dimension
    ):
        raise ArtifactRuntimeError(
            "Composite fitted-state feature dimension conflicts with the manifest DAG."
        )

    from pyimgano.inference.composite_runtime import CompositeArtifactRuntime

    return (
        CompositeArtifactRuntime(
            component_runtime=component_runtime,
            fitted_core=fitted_core,
            adapter=compose,
            adapter_id=adapter_id,
            runtime_info={
                "backend": "composite",
                "composition_adapter": {
                    "id": adapter_id,
                    "version": adapter_version,
                },
            },
        ),
        model_name,
    )


def _build_backend(
    manifest: Mapping[str, Any],
    *,
    staged: Any,
    device: str | None,
    providers: Any,
    session_options: Mapping[str, Any] | None,
    trust_checkpoint: bool,
) -> tuple[Any, str | None]:
    if str(manifest.get("layout", "")).strip().lower() == "composite":
        return _load_composite(
            manifest,
            staged=staged,
            device=device,
            providers=providers,
            session_options=session_options,
            trust_checkpoint=bool(trust_checkpoint),
        )
    runtime = _mapping(manifest.get("runtime"))
    backend = str(runtime.get("backend", "")).strip().lower()
    input_contract = _mapping(manifest.get("input_contract"))
    output_contract = _mapping(manifest.get("output_contract"))
    entrypoint = str(runtime.get("entrypoint", "")).strip()
    if backend == "onnxruntime":
        from pyimgano.inference.onnx_runtime import (
            OnnxArtifactRuntime,
            resolve_onnx_session_options,
        )

        selected_options = resolve_onnx_session_options(
            (
                _mapping(runtime.get("session_options"))
                if runtime.get("session_options") is not None
                else None
            ),
            session_options,
        )
        compatibility = _mapping(manifest.get("compatibility"))

        return (
            OnnxArtifactRuntime(
                staged.path_for(entrypoint),
                input_contract=input_contract,
                output_contract=output_contract,
                allowed_providers=runtime.get("allowed_providers"),
                verified_providers=runtime.get("verified_providers"),
                providers=providers,
                device=device,
                session_options=selected_options,
                expected_onnx_ir=compatibility.get("onnx_ir"),
                expected_onnx_opset=compatibility.get("onnx_opset"),
            ),
            None,
        )
    if backend == "torchscript":
        if providers is not None:
            raise ArtifactRuntimeError(
                "providers is only valid for provider-based artifact backends."
            )
        if session_options is not None:
            raise ArtifactRuntimeError("session_options is not supported by TorchScript artifacts.")
        from pyimgano.inference.torchscript_runtime import TorchScriptArtifactRuntime

        return (
            TorchScriptArtifactRuntime(
                staged.path_for(entrypoint),
                input_contract=input_contract,
                output_contract=output_contract,
                allowed_providers=runtime.get("allowed_providers"),
                verified_providers=runtime.get("verified_providers"),
                device=device,
                trust_checkpoint=bool(trust_checkpoint),
            ),
            None,
        )
    if backend == "openvino":
        if providers is not None:
            raise ArtifactRuntimeError(
                "providers is not supported by OpenVINO artifacts; use device."
            )
        if session_options is not None:
            raise ArtifactRuntimeError("session_options is not supported by OpenVINO artifacts.")
        from pyimgano.inference.openvino_runtime import OpenVINOArtifactRuntime

        return (
            OpenVINOArtifactRuntime(
                staged.path_for(entrypoint),
                input_contract=input_contract,
                output_contract=output_contract,
                allowed_providers=runtime.get("allowed_providers"),
                verified_providers=runtime.get("verified_providers"),
                device=device,
            ),
            None,
        )
    if backend == "pyimgano":
        return _load_native_or_composite(
            manifest,
            staged=staged,
            device=device,
            providers=providers,
            session_options=session_options,
            trust_checkpoint=trust_checkpoint,
        )
    raise ArtifactRuntimeError(f"Unsupported artifact runtime backend: {backend!r}")


def _apply_detector_policy(detector: Any, policy: Mapping[str, Any]) -> Any:
    adaptation = _mapping(policy.get("adaptation"))
    tiling = _mapping(adaptation.get("tiling"))
    if tiling.get("tile_size") is not None:
        from pyimgano.inference.tiling import TiledDetector

        detector = TiledDetector(
            detector=detector,
            tile_size=int(tiling["tile_size"]),
            stride=(int(tiling["stride"]) if tiling.get("stride") is not None else None),
            score_reduce=str(tiling.get("score_reduce", "max")),
            score_topk=float(tiling.get("score_topk", 0.1)),
            map_reduce=str(tiling.get("map_reduce", "max")),
        )
    preprocessing = _mapping(policy.get("preprocessing"))
    illumination = _mapping(preprocessing.get("illumination_contrast"))
    if illumination:
        from pyimgano.inference.preprocessing import (
            PreprocessingDetector,
            parse_illumination_contrast_knobs,
        )

        detector = PreprocessingDetector(
            detector=detector,
            illumination_contrast=parse_illumination_contrast_knobs(illumination),
        )
    return detector


def load_artifact(
    artifact: str | Path,
    *,
    category: str | None = None,
    format: str | None = None,
    backend: str | None = None,
    artifact_id: str | None = None,
    device: str | None = None,
    providers: Sequence[str | Mapping[str, Any]] | None = None,
    session_options: Mapping[str, Any] | None = None,
    trust_checkpoint: bool = False,
) -> ArtifactRuntime:
    """Load a verified artifact and return a detector-compatible runtime."""

    if artifact_id is not None and any(
        selector is not None for selector in (category, format, backend)
    ):
        raise ArtifactRuntimeError(
            "artifact_id is an exact selector and cannot be combined with "
            "category, format, or backend."
        )

    manifest_path = _resolve_artifact_source(
        artifact,
        category=category,
        artifact_format=format,
        backend=backend,
        artifact_id=artifact_id,
    )
    artifact_root = manifest_path.parent
    try:
        from pyimgano.artifacts import (
            load_artifact_manifest,
            stage_verified_artifact,
            validate_artifact_policy,
        )
    except ImportError as exc:  # pragma: no cover - transitional source checkout guard
        raise ArtifactRuntimeError(
            "Artifact manifest support is unavailable in this pyimgano build."
        ) from exc

    manifest = load_artifact_manifest(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ArtifactRuntimeError("load_artifact_manifest() returned a non-mapping value.")
    manifest = dict(manifest)
    model = _mapping(manifest.get("model"))
    runtime_contract = _mapping(manifest.get("runtime"))
    declared_category = model.get("category")
    if category is not None and str(category) != str(declared_category):
        raise ArtifactRuntimeError(
            f"Artifact category mismatch: requested={category!r}, manifest={declared_category!r}."
        )
    if backend is not None and str(runtime_contract.get("backend")) != str(backend):
        raise ArtifactRuntimeError(
            f"Artifact backend mismatch: requested={backend!r}, "
            f"manifest={runtime_contract.get('backend')!r}."
        )
    if artifact_id is not None and str(manifest.get("artifact_id")) != str(artifact_id):
        raise ArtifactRuntimeError(
            f"Artifact ID mismatch: requested={artifact_id!r}, manifest={manifest.get('artifact_id')!r}."
        )
    if format is not None:
        runtime_component = _component_by_role(manifest, "runtime_model")
        if runtime_component is not None:
            declared_format = _selector_format(runtime_component.get("format"))
        elif str(manifest.get("layout")) == "native_detector":
            declared_format = "native"
        else:
            declared_format = None
        if str(declared_format) != str(_selector_format(format)):
            raise ArtifactRuntimeError(
                f"Artifact format mismatch: requested={format!r}, manifest={declared_format!r}."
            )

    requires_executable_trust = str(runtime_contract.get("backend", "")).lower() == "torchscript"
    requires_executable_trust = requires_executable_trust or any(
        isinstance(component, Mapping)
        and str(component.get("serialization", "")) == "executable-trust-required"
        for component in manifest.get("components", [])
    )
    if requires_executable_trust and not trust_checkpoint:
        raise ArtifactRuntimeError(
            "Artifact requires executable deserialization. Reload only with "
            "trust_checkpoint=True after verifying provenance."
        )

    try:
        from pyimgano.artifacts.compatibility import preflight_artifact_compatibility

        compatibility_report = preflight_artifact_compatibility(manifest)
    except (ImportError, TypeError, ValueError) as exc:
        raise ArtifactRuntimeError(f"Artifact compatibility preflight failed: {exc}") from exc
    adapter = _validate_registered_adapter(manifest)

    stage_context = stage_verified_artifact(artifact_root, manifest)
    staged = stage_context.__enter__()
    closed = False

    def _cleanup() -> None:
        nonlocal closed
        if not closed:
            closed = True
            stage_context.__exit__(None, None, None)

    try:
        _validate_safe_state_bindings(manifest, staged=staged, adapter=adapter)
        policy_ref = _mapping(manifest.get("policy_ref"))
        policy_path = staged.path_for(str(policy_ref.get("path", "infer_config.json")))
        policy = _load_json_object(Path(policy_path))
        policy = validate_artifact_policy(policy, manifest_model=model or None)
        detector, model_name = _build_backend(
            manifest,
            staged=staged,
            device=device,
            providers=providers,
            session_options=session_options,
            trust_checkpoint=bool(trust_checkpoint),
        )
        detector = _apply_detector_policy(detector, policy)
        runtime = ArtifactRuntime(
            detector,
            manifest=manifest,
            infer_config=policy,
            artifact_root=artifact_root,
            model_name=model_name,
            runtime_info={
                "backend": runtime_contract.get("backend"),
                "compatibility": {
                    "pyimgano": compatibility_report.pyimgano_version,
                    "python": compatibility_report.python_version,
                    "platform": compatibility_report.platform_tag,
                    "runtime_versions": dict(compatibility_report.runtime_versions),
                },
            },
            cleanup=_cleanup,
        )
    except Exception:
        _cleanup()
        raise
    return runtime


__all__ = ["load_artifact"]
