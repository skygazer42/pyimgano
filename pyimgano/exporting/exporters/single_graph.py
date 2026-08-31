from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.artifacts.compatibility import current_platform_tag
from pyimgano.exporting.protocols import ExportAdapterProtocol
from pyimgano.exporting.types import (
    ArtifactFormat,
    ExportLayout,
    ExportStatus,
    ExportTargetKind,
    NativeExportContext,
    ProbeSpec,
)
from pyimgano.exporting.writer import ArtifactWriter


class SingleGraphExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphArtifactExportResult:
    artifact_root: Path
    manifest_path: Path
    graph_path: Path
    policy_path: Path
    manifest: Mapping[str, Any]


def _call_capability(
    adapter: ExportAdapterProtocol,
    format: ArtifactFormat,
    context: NativeExportContext,
) -> Any:
    effective = getattr(adapter, "effective_capability", None)
    if callable(effective):
        return effective(format, context=context)
    return adapter.declared_capability(format)


def _validate_capability(
    adapter: ExportAdapterProtocol,
    format: ArtifactFormat,
    context: NativeExportContext,
) -> None:
    capability = _call_capability(adapter, format, context)
    if (
        capability.status is not ExportStatus.SUPPORTED
        or capability.target_kind is not ExportTargetKind.ARTIFACT
        or capability.layout is not ExportLayout.SINGLE_GRAPH
        or not capability.supported
    ):
        raise SingleGraphExportError(
            "Model does not have effective fitted single-graph export support: "
            f"format={format.value!r}, reason={capability.reason_code!r}, "
            f"availability={capability.availability!s}."
        )
    validate = getattr(adapter, "validate_checkpoint_contract", None)
    if not callable(validate):
        raise SingleGraphExportError(
            f"Adapter {adapter.adapter_id!r} does not validate checkpoint contracts."
        )
    validate(context.checkpoint_contract, context=context)


def _model_payload(context: NativeExportContext) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "registry_name": str(context.model_name),
        "constructor_kwargs": dict(context.model_kwargs),
    }
    if context.category is not None:
        payload["category"] = str(context.category)
    return payload


def _runtime_payload(
    adapter: ExportAdapterProtocol,
    format: ArtifactFormat,
    context: NativeExportContext,
    *,
    entrypoint: str,
) -> dict[str, Any]:
    build = getattr(adapter, "build_runtime_spec", None)
    if not callable(build):
        raise SingleGraphExportError(
            f"Adapter {adapter.adapter_id!r} does not declare a runtime contract."
        )
    runtime = dict(build(format=format, context=context) or {})
    expected_backend = {
        ArtifactFormat.ONNX: "onnxruntime",
        ArtifactFormat.TORCHSCRIPT: "torchscript",
        ArtifactFormat.OPENVINO: "openvino",
    }[format]
    if str(runtime.get("backend", "")) != expected_backend:
        raise SingleGraphExportError(
            f"Adapter runtime backend must be {expected_backend!r} for {format.value}."
        )
    runtime["entrypoint"] = str(entrypoint)
    return runtime


def _compatibility_payload(
    adapter: ExportAdapterProtocol,
    format: ArtifactFormat,
    context: NativeExportContext,
    *,
    onnx_opset: int | None = None,
    onnx_ir: int | None = None,
) -> dict[str, Any]:
    runtime_versions = {
        ArtifactFormat.ONNX: {"onnxruntime": ">=1.17,<2"},
        ArtifactFormat.TORCHSCRIPT: {"torch": ">=1.9"},
        ArtifactFormat.OPENVINO: {"openvino": ">=2023,<2027"},
    }[format]
    if format is ArtifactFormat.ONNX:
        if onnx_opset is None or onnx_ir is None:
            raise SingleGraphExportError(
                "ONNX compatibility requires certified IR and default-opset metadata."
            )
        from pyimgano.artifacts.compatibility import onnxruntime_requirement_for_graph

        runtime_versions = {
            "onnxruntime": onnxruntime_requirement_for_graph(
                ir_version=onnx_ir,
                default_opset=onnx_opset,
            )
        }
    payload: dict[str, Any] = {
        "pyimgano": ">=0.10,<0.11",
        "python": ">=3.9,<3.13",
        "platforms": [current_platform_tag()],
        "runtime_versions": runtime_versions,
        "adapter": {
            "id": str(adapter.adapter_id),
            "version": int(adapter.adapter_version),
        },
        "codecs": [],
    }
    payload.update(dict(context.compatibility))
    # Executable identities are adapter-owned and cannot be overridden by hints.
    payload["adapter"] = {
        "id": str(adapter.adapter_id),
        "version": int(adapter.adapter_version),
    }
    payload["codecs"] = []
    payload["platforms"] = [current_platform_tag()]
    payload["runtime_versions"] = runtime_versions
    if onnx_opset is not None:
        payload["onnx_opset"] = int(onnx_opset)
    if onnx_ir is not None:
        payload["onnx_ir"] = int(onnx_ir)
    return payload


def _contracts(adapter: Any, detector: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    input_builder = getattr(adapter, "build_input_contract", None)
    output_builder = getattr(adapter, "build_output_contract", None)
    if not callable(input_builder) or not callable(output_builder):
        raise SingleGraphExportError(
            "Single-graph adapters must declare input and output contracts."
        )
    return dict(input_builder(detector)), dict(output_builder())


def _reference_probe(adapter: Any, detector: Any) -> tuple[ProbeSpec, dict[str, np.ndarray]]:
    build_probe = getattr(adapter, "build_probe_spec", None)
    evaluate = getattr(adapter, "evaluate_probe", None)
    if not callable(build_probe) or not callable(evaluate):
        raise SingleGraphExportError("Single-graph adapters must provide fixed reference probes.")
    spec = build_probe(detector)
    reference = dict(evaluate(detector, spec))
    for name in ("score", "anomaly_map"):
        if name not in reference:
            raise SingleGraphExportError(f"Reference probe is missing output {name!r}.")
        reference[name] = np.asarray(reference[name], dtype=np.float32)
    return spec, reference


def _runtime_outputs(runtime: Any, inputs: Sequence[Any]) -> dict[str, np.ndarray]:
    scores, maps = runtime.score_and_maps(list(inputs), include_maps=True)
    if maps is None:
        raise SingleGraphExportError("Exported graph runtime did not return anomaly maps.")
    return {
        "score": np.asarray(scores, dtype=np.float32),
        "anomaly_map": np.asarray(maps, dtype=np.float32),
    }


def _compare_outputs(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    passed = True
    for name in ("score", "anomaly_map"):
        expected = np.asarray(reference[name], dtype=np.float32)
        observed = np.asarray(actual[name], dtype=np.float32)
        if expected.shape != observed.shape:
            outputs[name] = {
                "passed": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(observed.shape),
            }
            passed = False
            continue
        absolute = np.abs(observed - expected)
        relative = absolute / np.maximum(np.abs(expected), np.float32(1e-12))
        output_passed = bool(
            np.allclose(
                observed,
                expected,
                atol=float(absolute_tolerance),
                rtol=float(relative_tolerance),
                equal_nan=False,
            )
        )
        outputs[name] = {
            "passed": output_passed,
            "shape": list(expected.shape),
            "max_absolute_error": float(absolute.max(initial=0.0)),
            "max_relative_error": float(relative.max(initial=0.0)),
        }
        passed = passed and output_passed
    return {"passed": bool(passed), "outputs": outputs}


def _verify_runtime(
    runtime: Any,
    *,
    probe: ProbeSpec,
    reference: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    all_outputs = _runtime_outputs(runtime, probe.inputs)
    report = _compare_outputs(
        reference,
        all_outputs,
        absolute_tolerance=probe.absolute_tolerance,
        relative_tolerance=probe.relative_tolerance,
    )
    # Exercise the dynamic batch/tail path independently of the N-sample call.
    tail_outputs = _runtime_outputs(runtime, probe.inputs[-1:])
    tail_reference = {
        name: np.asarray(value[-1:], dtype=np.float32) for name, value in reference.items()
    }
    tail = _compare_outputs(
        tail_reference,
        tail_outputs,
        absolute_tolerance=probe.absolute_tolerance,
        relative_tolerance=probe.relative_tolerance,
    )
    report["tail_batch"] = tail
    report["passed"] = bool(report["passed"] and tail["passed"])
    report["sample_count"] = len(probe.inputs)
    report["batch_sizes_verified"] = sorted({1, len(probe.inputs)})
    report["absolute_tolerance"] = float(probe.absolute_tolerance)
    report["relative_tolerance"] = float(probe.relative_tolerance)
    if not bool(report["passed"]):
        raise SingleGraphExportError(f"Exported graph failed reference parity: {report!r}.")
    return report


def _attachment_metadata(writer: ArtifactWriter, relative_path: str) -> dict[str, Any]:
    metadata = writer.component_metadata(
        relative_path,
        component_id="verification",
        role="verification",
        format="json",
        serialization="safe-data",
    )
    return {
        "path": str(relative_path),
        "size_bytes": int(metadata["size_bytes"]),
        "sha256": str(metadata["sha256"]),
    }


def export_single_graph(
    detector: Any,
    *,
    format: ArtifactFormat,
    context: NativeExportContext,
    out: str | Path,
    adapter: ExportAdapterProtocol,
    overwrite: bool = False,
) -> GraphArtifactExportResult:
    """Export, execute, compare, manifest, and atomically publish a fitted graph."""

    if format not in {
        ArtifactFormat.ONNX,
        ArtifactFormat.TORCHSCRIPT,
        ArtifactFormat.OPENVINO,
    }:
        raise SingleGraphExportError(f"{format.value!r} is not a single-graph format.")
    _validate_capability(adapter, format, context)
    build_graph = getattr(adapter, "build_graph_export_spec", None)
    if not callable(build_graph):
        raise SingleGraphExportError(
            f"Adapter {adapter.adapter_id!r} does not implement graph export."
        )
    graph_spec = build_graph(detector, format=format, context=context)
    input_contract, output_contract = _contracts(adapter, detector)
    probe, reference = _reference_probe(adapter, detector)

    output_root = Path(out)
    onnx_opset: int | None = None
    onnx_ir: int | None = None
    source_onnx_report: dict[str, Any] | None = None
    with ArtifactWriter(output_root, overwrite=overwrite) as writer:
        components: list[dict[str, Any]] = []
        if format is ArtifactFormat.ONNX:
            from pyimgano.exporting.exporters.onnx import export_onnx_graph
            from pyimgano.inference.onnx_runtime import OnnxArtifactRuntime

            relative = "model/detector.onnx"
            info = export_onnx_graph(graph_spec, writer.path_for(relative))
            onnx_opset, onnx_ir = info.opset, info.ir_version
            runtime = OnnxArtifactRuntime(
                info.path,
                input_contract=input_contract,
                output_contract=output_contract,
                allowed_providers=[{"name": "CPUExecutionProvider", "options": {}}],
                verified_providers=[{"name": "CPUExecutionProvider", "options": {}}],
            )
            parity = _verify_runtime(runtime, probe=probe, reference=reference)
            components.append(
                writer.component_metadata(
                    relative,
                    component_id="detector-graph",
                    role="runtime_model",
                    format="onnx",
                    serialization="onnx",
                )
            )
        elif format is ArtifactFormat.TORCHSCRIPT:
            from pyimgano.exporting.exporters.torchscript import export_torchscript_graph
            from pyimgano.inference.torchscript_runtime import TorchScriptArtifactRuntime

            relative = "model/detector.pt"
            info = export_torchscript_graph(graph_spec, writer.path_for(relative))
            runtime = TorchScriptArtifactRuntime(
                info.path,
                input_contract=input_contract,
                output_contract=output_contract,
                allowed_providers=[{"name": "CPU", "options": {}}],
                verified_providers=[{"name": "CPU", "options": {}}],
                device="cpu",
                trust_checkpoint=True,
            )
            parity = _verify_runtime(runtime, probe=probe, reference=reference)
            components.append(
                writer.component_metadata(
                    relative,
                    component_id="detector-graph",
                    role="runtime_model",
                    format="torchscript",
                    serialization="executable-trust-required",
                )
            )
        else:
            from pyimgano.exporting.exporters.onnx import export_onnx_graph
            from pyimgano.exporting.exporters.openvino import (
                convert_verified_onnx_to_openvino,
            )
            from pyimgano.inference.onnx_runtime import OnnxArtifactRuntime
            from pyimgano.inference.openvino_runtime import OpenVINOArtifactRuntime

            relative = "model/detector.xml"
            with tempfile.TemporaryDirectory(prefix="pyimgano-openvino-source-") as temporary:
                onnx_info = export_onnx_graph(
                    graph_spec,
                    Path(temporary) / "detector.onnx",
                )
                onnx_opset, onnx_ir = onnx_info.opset, onnx_info.ir_version
                onnx_runtime = OnnxArtifactRuntime(
                    onnx_info.path,
                    input_contract=input_contract,
                    output_contract=output_contract,
                    allowed_providers=[{"name": "CPUExecutionProvider", "options": {}}],
                    verified_providers=[{"name": "CPUExecutionProvider", "options": {}}],
                )
                source_onnx_report = _verify_runtime(
                    onnx_runtime,
                    probe=probe,
                    reference=reference,
                )
                info = convert_verified_onnx_to_openvino(
                    onnx_info.path,
                    writer.path_for(relative),
                )
            runtime = OpenVINOArtifactRuntime(
                info.xml_path,
                input_contract=input_contract,
                output_contract=output_contract,
                allowed_providers=[{"name": "CPU", "options": {}}],
                verified_providers=[{"name": "CPU", "options": {}}],
                device="CPU",
            )
            parity = _verify_runtime(runtime, probe=probe, reference=reference)
            components.extend(
                [
                    writer.component_metadata(
                        relative,
                        component_id="detector-graph",
                        role="runtime_model",
                        format="openvino-ir",
                        serialization="openvino-ir",
                    ),
                    writer.component_metadata(
                        "model/detector.bin",
                        component_id="detector-weights",
                        role="openvino_weights",
                        format="openvino-weights",
                        serialization="safe-data",
                    ),
                ]
            )

        verification_values = dict(context.verification)
        verification_level = str(verification_values.pop("level", "reference_parity"))
        report: dict[str, Any] = dict(verification_values)
        report.update(
            {
                "format": format.value,
                "adapter": {
                    "id": str(adapter.adapter_id),
                    "version": int(adapter.adapter_version),
                },
                "checkpoint_sha256": context.checkpoint_contract.sha256,
                "reference_backend": "pyimgano",
                "target_backend": _runtime_payload(
                    adapter,
                    format,
                    context,
                    entrypoint=relative,
                )["backend"],
                "parity": parity,
                "graph_outputs": list(graph_spec.output_names),
                "output_semantics": dict(graph_spec.output_semantics),
                "mandatory": True,
            }
        )
        if source_onnx_report is not None:
            report["verified_onnx_source_parity"] = source_onnx_report
        verification_path = writer.write_json("verification/parity.json", report)
        if not verification_path.is_file():  # pragma: no cover - writer invariant
            raise SingleGraphExportError("Failed to write graph parity report.")

        runtime_payload = _runtime_payload(
            adapter,
            format,
            context,
            entrypoint=relative,
        )
        manifest_payload: dict[str, Any] = {
            "schema_family": "pyimgano-artifact",
            "schema_version": 1,
            "layout": ExportLayout.SINGLE_GRAPH.value,
            "model": _model_payload(context),
            "runtime": runtime_payload,
            "input_contract": input_contract,
            "output_contract": output_contract,
            "components": components,
            "policy_ref": {"path": "infer_config.json"},
            "compatibility": _compatibility_payload(
                adapter,
                format,
                context,
                onnx_opset=onnx_opset,
                onnx_ir=onnx_ir,
            ),
            "verification": {
                "level": verification_level,
                "reference_backend": "pyimgano",
                "report": _attachment_metadata(writer, "verification/parity.json"),
            },
        }
        manifest_path, manifest = writer.finalize(
            manifest_payload,
            policy=dict(context.policy),
        )

    return GraphArtifactExportResult(
        artifact_root=output_root,
        manifest_path=manifest_path,
        graph_path=output_root / relative,
        policy_path=output_root / "infer_config.json",
        manifest=manifest,
    )


__all__ = [
    "GraphArtifactExportResult",
    "SingleGraphExportError",
    "export_single_graph",
]
