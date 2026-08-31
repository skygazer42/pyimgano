from __future__ import annotations

import copy
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import numpy as np

from pyimgano.artifacts.compatibility import current_platform_tag
from pyimgano.exporting.protocols import ExportAdapterProtocol
from pyimgano.exporting.state_codec import get_state_codec, save_fitted_state
from pyimgano.exporting.types import (
    ArtifactFormat,
    ExportLayout,
    ExportStatus,
    ExportTargetKind,
    NativeExportContext,
    ProbeSpec,
)
from pyimgano.exporting.writer import ArtifactWriter


class CompositeExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompositeArtifactExportResult:
    artifact_root: Path
    manifest_path: Path
    graph_path: Path
    state_path: Path
    policy_path: Path
    manifest: Mapping[str, Any]


def _validate_capability(
    adapter: ExportAdapterProtocol,
    format: ArtifactFormat,
    context: NativeExportContext,
) -> None:
    effective = getattr(adapter, "effective_capability", None)
    capability = (
        effective(format, context=context)
        if callable(effective)
        else adapter.declared_capability(format)
    )
    if (
        capability.status is not ExportStatus.SUPPORTED
        or capability.target_kind is not ExportTargetKind.ARTIFACT
        or capability.layout is not ExportLayout.COMPOSITE
        or not capability.supported
    ):
        raise CompositeExportError(
            "Model does not have effective fitted composite export support: "
            f"format={format.value!r}, reason={capability.reason_code!r}, "
            f"availability={capability.availability!s}."
        )
    validate = getattr(adapter, "validate_checkpoint_contract", None)
    if not callable(validate):
        raise CompositeExportError(
            f"Adapter {adapter.adapter_id!r} does not validate checkpoint contracts."
        )
    validate(context.checkpoint_contract, context=context)


def _safe_onnx_location(value: str) -> str:
    location = str(value)
    if (
        not location
        or location != location.strip()
        or unicodedata.normalize("NFC", location) != location
        or "\x00" in location
        or "\\" in location
        or "//" in location
        or location.endswith("/")
    ):
        raise CompositeExportError(f"Unsafe ONNX external-data location: {location!r}.")
    posix = PurePosixPath(location)
    windows = PureWindowsPath(location)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise CompositeExportError(f"Unsafe ONNX external-data location: {location!r}.")
    if any(part in {"", ".", ".."} for part in location.split("/")):
        raise CompositeExportError(f"Unsafe ONNX external-data location: {location!r}.")
    return location


def _copy_embedding_graph(
    writer: ArtifactWriter,
    *,
    source_path: Path,
    format: ArtifactFormat,
) -> tuple[str, list[dict[str, Any]], int | None, int | None]:
    if format is ArtifactFormat.ONNX:
        relative = "model/embedding.onnx"
        target = writer.copy_file(source_path, relative)
        try:
            import onnx

            from pyimgano.artifacts.onnx_external_data import external_data_locations

            graph = onnx.load_model(str(target), load_external_data=False)
            locations = [_safe_onnx_location(item) for item in external_data_locations(graph)]
        except CompositeExportError:
            raise
        except Exception as exc:  # noqa: BLE001 - copied graph is a trust boundary
            raise CompositeExportError(
                f"Cannot inspect the copied ONNX embedding graph: {exc}"
            ) from exc
        if any(item.casefold() == "embedding.onnx" for item in locations):
            raise CompositeExportError(
                "ONNX external data must not overwrite the embedding graph entrypoint."
            )
        components = [
            writer.component_metadata(
                relative,
                component_id="embedding-graph",
                role="runtime_model",
                format="onnx",
                serialization="onnx",
            )
        ]
        for index, location in enumerate(locations):
            source_dependency = source_path.parent.joinpath(*location.split("/"))
            dependency_relative = f"model/{location}"
            writer.copy_file(source_dependency, dependency_relative)
            components.append(
                writer.component_metadata(
                    dependency_relative,
                    component_id=f"embedding-external-{index}",
                    role="external_data",
                    format="onnx-external-data",
                    serialization="safe-data",
                )
            )
        default_opsets = [
            int(item.version)
            for item in graph.opset_import
            if str(item.domain or "") in {"", "ai.onnx"}
        ]
        onnx_opset = max(default_opsets) if default_opsets else None
        return relative, components, onnx_opset, int(graph.ir_version)

    if format is ArtifactFormat.TORCHSCRIPT:
        relative = "model/embedding.pt"
        writer.copy_file(source_path, relative)
        return (
            relative,
            [
                writer.component_metadata(
                    relative,
                    component_id="embedding-graph",
                    role="runtime_model",
                    format="torchscript",
                    serialization="executable-trust-required",
                )
            ],
            None,
            None,
        )
    raise CompositeExportError(f"Composite embedding format {format.value!r} is not supported.")


def _build_component_runtime(
    *,
    format: ArtifactFormat,
    graph_path: Path,
    component_spec: Any,
) -> Any:
    if format is ArtifactFormat.ONNX:
        from pyimgano.inference.composite_runtime import OnnxEmbeddingComponentRuntime

        return OnnxEmbeddingComponentRuntime(
            graph_path,
            input_contract=component_spec.input_contract,
            output_contract=component_spec.output_contract,
            batch_size=int(component_spec.batch_size),
            allowed_providers=list(component_spec.allowed_providers),
            verified_providers=list(component_spec.verified_providers),
            providers=list(component_spec.verified_providers),
            session_options=dict(component_spec.session_options),
        )
    if format is ArtifactFormat.TORCHSCRIPT:
        from pyimgano.inference.composite_runtime import (
            TorchScriptEmbeddingComponentRuntime,
        )

        return TorchScriptEmbeddingComponentRuntime(
            graph_path,
            input_contract=component_spec.input_contract,
            output_contract=component_spec.output_contract,
            batch_size=int(component_spec.batch_size),
            device="cpu",
            trust_checkpoint=True,
        )
    raise CompositeExportError(f"Unsupported composite format: {format.value!r}.")


def _reference_probe(
    adapter: Any,
    detector: Any,
    context: NativeExportContext,
) -> tuple[ProbeSpec, np.ndarray]:
    build_probe = getattr(adapter, "build_probe_spec", None)
    evaluate = getattr(adapter, "evaluate_probe", None)
    if not callable(build_probe) or not callable(evaluate):
        raise CompositeExportError(
            "Composite adapters must provide deterministic reference probes."
        )
    probe = build_probe(detector, context=context)
    reference = dict(evaluate(detector, probe))
    if set(reference) != {"score"}:
        raise CompositeExportError(
            "ECOD composite reference probe must contain exactly the score output."
        )
    scores = np.asarray(reference["score"], dtype=np.float64).reshape(-1)
    if scores.shape != (len(probe.inputs),) or not np.isfinite(scores).all():
        raise CompositeExportError("Composite reference probe returned invalid scores.")
    return probe, scores


def _compare_scores(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    probe: ProbeSpec,
) -> dict[str, Any]:
    left = np.asarray(expected, dtype=np.float64).reshape(-1)
    right = np.asarray(observed, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        return {
            "passed": False,
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
        }
    absolute = np.abs(right - left)
    relative = absolute / np.maximum(np.abs(left), 1e-12)
    passed = bool(
        np.allclose(
            right,
            left,
            atol=float(probe.absolute_tolerance),
            rtol=float(probe.relative_tolerance),
            equal_nan=False,
        )
    )
    return {
        "passed": passed,
        "shape": list(left.shape),
        "max_absolute_error": float(absolute.max(initial=0.0)),
        "max_relative_error": float(relative.max(initial=0.0)),
    }


def _verify_runtime(
    runtime: Any,
    *,
    probe: ProbeSpec,
    reference: np.ndarray,
) -> dict[str, Any]:
    scores, maps = runtime.score_and_maps(probe.inputs, include_maps=True)
    if maps is not None:
        raise CompositeExportError("ECOD composite unexpectedly returned anomaly maps.")
    report = _compare_scores(reference, scores, probe=probe)
    tail_scores, tail_maps = runtime.score_and_maps(probe.inputs[-1:], include_maps=True)
    if tail_maps is not None:
        raise CompositeExportError("ECOD composite tail probe returned anomaly maps.")
    tail = _compare_scores(reference[-1:], tail_scores, probe=probe)
    report["tail_batch"] = tail
    report["passed"] = bool(report["passed"] and tail["passed"])
    report["sample_count"] = len(probe.inputs)
    report["batch_sizes_verified"] = sorted({1, len(probe.inputs)})
    report["absolute_tolerance"] = float(probe.absolute_tolerance)
    report["relative_tolerance"] = float(probe.relative_tolerance)
    if not report["passed"]:
        raise CompositeExportError(
            f"Composite artifact failed mandatory reference parity: {report!r}."
        )
    return report


def _model_payload(
    context: NativeExportContext,
    *,
    constructor_kwargs: Mapping[str, Any],
    graph_relative: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "registry_name": str(context.model_name),
        "constructor_kwargs": copy.deepcopy(dict(constructor_kwargs)),
        "asset_bindings": {
            "embedding_kwargs.checkpoint_path": str(graph_relative),
        },
    }
    if context.category is not None:
        payload["category"] = str(context.category)
    return payload


def _artifact_policy(
    context: NativeExportContext,
    *,
    model_payload: Mapping[str, Any],
) -> dict[str, Any]:
    policy = copy.deepcopy(dict(context.policy))
    policy["model"] = copy.deepcopy(dict(model_payload))
    return policy


def _runtime_payload(
    adapter: Any, format: ArtifactFormat, context: NativeExportContext
) -> dict[str, Any]:
    build = getattr(adapter, "build_runtime_spec", None)
    if not callable(build):
        raise CompositeExportError("Composite adapter does not declare a runtime contract.")
    runtime = dict(build(format=format, context=context) or {})
    if runtime.get("backend") != "pyimgano":
        raise CompositeExportError(
            "Composite artifact orchestration requires runtime.backend='pyimgano'."
        )
    if "entrypoint" in runtime:
        raise CompositeExportError("Composite runtime must use DAG bindings, not entrypoint.")
    return runtime


def _compatibility_payload(
    adapter: Any,
    format: ArtifactFormat,
    context: NativeExportContext,
    *,
    codec_id: str,
    codec_version: int,
    onnx_opset: int | None,
    onnx_ir: int | None,
) -> dict[str, Any]:
    if format is ArtifactFormat.ONNX:
        if onnx_opset is None or onnx_ir is None:
            raise CompositeExportError(
                "ONNX composite compatibility requires certified IR and opset metadata."
            )
        from pyimgano.artifacts.compatibility import onnxruntime_requirement_for_graph

        runtime_versions = {
            "onnxruntime": onnxruntime_requirement_for_graph(
                ir_version=onnx_ir,
                default_opset=onnx_opset,
            )
        }
    else:
        runtime_versions = {"torch": ">=1.9"}
    payload: dict[str, Any] = {
        "pyimgano": ">=0.10,<0.11",
        "python": ">=3.9,<3.13",
        "platforms": [current_platform_tag()],
        "runtime_versions": runtime_versions,
        "adapter": {
            "id": str(adapter.adapter_id),
            "version": int(adapter.adapter_version),
        },
        "codecs": [{"id": str(codec_id), "version": int(codec_version)}],
    }
    payload.update(copy.deepcopy(dict(context.compatibility)))
    payload["platforms"] = [current_platform_tag()]
    payload["runtime_versions"] = runtime_versions
    payload["adapter"] = {
        "id": str(adapter.adapter_id),
        "version": int(adapter.adapter_version),
    }
    payload["codecs"] = [{"id": str(codec_id), "version": int(codec_version)}]
    if onnx_opset is not None:
        payload["onnx_opset"] = int(onnx_opset)
    if onnx_ir is not None:
        payload["onnx_ir"] = int(onnx_ir)
    return payload


def _attachment_metadata(writer: ArtifactWriter, relative_path: str) -> dict[str, Any]:
    value = writer.component_metadata(
        relative_path,
        component_id="verification",
        role="verification",
        format="json",
        serialization="safe-data",
    )
    return {
        "path": str(relative_path),
        "size_bytes": int(value["size_bytes"]),
        "sha256": str(value["sha256"]),
    }


def export_composite(
    detector: Any,
    *,
    format: ArtifactFormat,
    context: NativeExportContext,
    out: str | Path,
    adapter: ExportAdapterProtocol,
    overwrite: bool = False,
) -> CompositeArtifactExportResult:
    """Publish an exact embedding graph plus complete safe fitted-core state."""

    if format not in {ArtifactFormat.ONNX, ArtifactFormat.TORCHSCRIPT}:
        raise CompositeExportError(f"{format.value!r} is not an ECOD composite embedding format.")
    _validate_capability(adapter, format, context)
    validate_binding = getattr(adapter, "validate_checkpoint_source_binding", None)
    if not callable(validate_binding):
        raise CompositeExportError(
            "Composite adapter does not bind checkpoints to exact source components."
        )
    validate_binding(detector, context.checkpoint_contract, context=context)

    build_spec = getattr(adapter, "build_component_export_spec", None)
    if not callable(build_spec):
        raise CompositeExportError("Composite adapter does not declare its components.")
    component_spec = build_spec(detector, format=format, context=context)
    if component_spec.format is not format:
        raise CompositeExportError("Embedding component format changed during inspection.")
    build_fingerprint = getattr(adapter, "build_checkpoint_fingerprint_payload", None)
    if not callable(build_fingerprint):
        raise CompositeExportError(
            "Composite adapter does not expose its component fingerprint payload."
        )
    binding_payload = build_fingerprint(
        detector,
        context={"model_name": context.model_name},
    )
    if not isinstance(binding_payload, Mapping):
        raise CompositeExportError("Composite component fingerprint must be a mapping.")
    from pyimgano.artifacts import canonical_json_bytes

    binding_identity = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(dict(binding_payload))).hexdigest()
    )
    if binding_identity != context.checkpoint_contract.model_config_fingerprint:
        raise CompositeExportError(
            "Certified checkpoint is not bound to the current component closure."
        )
    expected_graph = binding_payload.get("embedding_graph")
    if not isinstance(expected_graph, Mapping):
        raise CompositeExportError("Composite fingerprint is missing embedding_graph.")

    codec_id = str(getattr(adapter, "state_codec_id", "") or "").strip()
    codec_version = int(getattr(adapter, "state_codec_version", 0) or 0)
    codec = get_state_codec(codec_id, codec_version)
    if (
        codec_id != context.checkpoint_contract.codec_id
        or codec_version != int(context.checkpoint_contract.codec_version or 0)
        or int(codec.state_schema_version)
        != int(context.checkpoint_contract.state_schema_version or 0)
    ):
        raise CompositeExportError(
            "Composite fitted-core codec does not match the certified checkpoint."
        )

    probe, reference = _reference_probe(adapter, detector, context)
    output_root = Path(out)
    graph_relative = ""
    onnx_opset: int | None = None
    onnx_ir: int | None = None
    with ArtifactWriter(output_root, overwrite=overwrite) as writer:
        graph_relative, components, onnx_opset, onnx_ir = _copy_embedding_graph(
            writer,
            source_path=Path(component_spec.source_path),
            format=format,
        )
        graph_component = next(
            (item for item in components if item.get("id") == "embedding-graph"),
            None,
        )
        if graph_component is None or (
            int(graph_component["size_bytes"]),
            str(graph_component["sha256"]),
        ) != (
            int(expected_graph.get("size_bytes", -1)),
            str(expected_graph.get("sha256", "")),
        ):
            raise CompositeExportError(
                "Securely copied embedding graph differs from the certified component."
            )
        expected_external_raw = expected_graph.get("external_data", [])
        if not isinstance(expected_external_raw, list):
            raise CompositeExportError("Composite fingerprint external_data must be a list.")
        expected_external = {
            str(item.get("location", "")): (
                int(item.get("size_bytes", -1)),
                str(item.get("sha256", "")),
            )
            for item in expected_external_raw
            if isinstance(item, Mapping)
        }
        actual_external = {
            str(item["path"])[len("model/") :]: (
                int(item["size_bytes"]),
                str(item["sha256"]),
            )
            for item in components
            if item.get("role") == "external_data"
        }
        if actual_external != expected_external:
            raise CompositeExportError(
                "Securely copied ONNX external-data closure differs from certification."
            )
        state_relative = "state/core.pyim"
        state_path = writer.path_for(state_relative)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_state(
            detector,
            state_path,
            model_name=context.model_name,
            checkpoint_contract=context.checkpoint_contract,
            codec_id=codec_id,
        )
        components.append(
            writer.component_metadata(
                state_relative,
                component_id="fitted-core",
                role="trained_state",
                format="pyimgano-state",
                serialization="safe-data",
            )
        )

        component_runtime = _build_component_runtime(
            format=format,
            graph_path=writer.path_for(graph_relative),
            component_spec=component_spec,
        )
        load_core = getattr(adapter, "load_composite_core", None)
        compose = getattr(adapter, "compose", None)
        if not callable(load_core) or not callable(compose):
            raise CompositeExportError(
                "Composite adapter must provide safe core loading and composition."
            )
        fitted_core = load_core(
            state_path,
            model_name=context.model_name,
            codec_id=codec_id,
            codec_version=codec_version,
        )
        from pyimgano.inference.composite_runtime import CompositeArtifactRuntime

        staged_runtime = CompositeArtifactRuntime(
            component_runtime=component_runtime,
            fitted_core=fitted_core,
            adapter=compose,
            adapter_id=str(adapter.adapter_id),
        )
        parity = _verify_runtime(staged_runtime, probe=probe, reference=reference)

        verification_values = copy.deepcopy(dict(context.verification))
        verification_level = str(verification_values.pop("level", "reference_parity"))
        report: dict[str, Any] = dict(verification_values)
        report.update(
            {
                "format": format.value,
                "adapter": {
                    "id": str(adapter.adapter_id),
                    "version": int(adapter.adapter_version),
                },
                "codec": {"id": codec_id, "version": codec_version},
                "checkpoint_sha256": context.checkpoint_contract.sha256,
                "reference_backend": "pyimgano",
                "target_backend": (
                    "onnxruntime" if format is ArtifactFormat.ONNX else "torchscript"
                ),
                "parity": parity,
                "mandatory": True,
                "source_embedding_copied_exactly": True,
            }
        )
        writer.write_json("verification/parity.json", report)

        model_payload = _model_payload(
            context,
            constructor_kwargs=component_spec.constructor_kwargs,
            graph_relative=graph_relative,
        )
        policy = _artifact_policy(context, model_payload=model_payload)
        runtime_payload = _runtime_payload(adapter, format, context)
        child_backend = "onnxruntime" if format is ArtifactFormat.ONNX else "torchscript"
        composition = {
            "nodes": [
                {
                    "id": "embedding",
                    "component": "embedding-graph",
                    "depends_on": [],
                    "operation": "embedding",
                    "runtime": {
                        "backend": child_backend,
                        "allowed_providers": list(component_spec.allowed_providers),
                        "verified_providers": list(component_spec.verified_providers),
                        **(
                            {"session_options": dict(component_spec.session_options)}
                            if format is ArtifactFormat.ONNX and component_spec.session_options
                            else {}
                        ),
                    },
                    "input_contract": dict(component_spec.input_contract),
                    "output_contract": dict(component_spec.output_contract),
                    "batch_size": int(component_spec.batch_size),
                },
                {
                    "id": "core",
                    "component": "fitted-core",
                    "depends_on": ["embedding"],
                    "operation": "fitted_core",
                    "codec": {"id": codec_id, "version": codec_version},
                    "state_model_name": str(context.model_name),
                    "feature_dimension": int(component_spec.feature_dimension),
                },
            ],
            "bindings": {
                "input": "embedding",
                "features": "embedding",
                "score": "core",
            },
        }
        manifest_payload: dict[str, Any] = {
            "schema_family": "pyimgano-artifact",
            "schema_version": 1,
            "layout": ExportLayout.COMPOSITE.value,
            "model": model_payload,
            "runtime": runtime_payload,
            "input_contract": dict(component_spec.input_contract),
            "output_contract": dict(adapter.build_output_contract()),
            "components": components,
            "composition": composition,
            "policy_ref": {"path": "infer_config.json"},
            "compatibility": _compatibility_payload(
                adapter,
                format,
                context,
                codec_id=codec_id,
                codec_version=codec_version,
                onnx_opset=onnx_opset,
                onnx_ir=onnx_ir,
            ),
            "verification": {
                "level": verification_level,
                "reference_backend": "pyimgano",
                "report": _attachment_metadata(writer, "verification/parity.json"),
            },
        }
        manifest_path, manifest = writer.finalize(manifest_payload, policy=policy)

    return CompositeArtifactExportResult(
        artifact_root=output_root,
        manifest_path=manifest_path,
        graph_path=output_root / graph_relative,
        state_path=output_root / "state" / "core.pyim",
        policy_path=output_root / "infer_config.json",
        manifest=manifest,
    )


__all__ = [
    "CompositeArtifactExportResult",
    "CompositeExportError",
    "export_composite",
]
