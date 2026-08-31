from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyimgano.exporting.types import GraphExportSpec

DEFAULT_ONNX_OPSET = 17


class OnnxGraphExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class OnnxGraphInfo:
    path: Path
    opset: int
    ir_version: int


def export_onnx_graph(
    spec: GraphExportSpec,
    path: str | Path,
    *,
    opset: int = DEFAULT_ONNX_OPSET,
) -> OnnxGraphInfo:
    """Serialize and structurally verify one full-detector ONNX graph."""

    if len(spec.example_inputs) != 1 or len(spec.input_names) != 1:
        raise OnnxGraphExportError("ONNX single-graph export requires exactly one input.")
    if tuple(spec.output_names) != ("score", "anomaly_map"):
        raise OnnxGraphExportError(
            "Certified autoencoder ONNX export requires score and anomaly_map outputs."
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - capability gate
        raise OnnxGraphExportError(
            "ONNX export requires torch; install pyimgano[onnx-export]."
        ) from exc

    export_parameters = inspect.signature(torch.onnx.export).parameters
    kwargs: dict[str, Any] = {
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
        "opset_version": int(opset),
        "dynamic_axes": {
            str(name): {int(axis): str(label) for axis, label in axes.items()}
            for name, axes in spec.dynamic_axes.items()
        },
        "do_constant_folding": True,
        "export_params": True,
    }
    # The legacy path remains the stable way to request opset 17 across the
    # supported Torch range.  Newer dynamo exporters require different dynamic
    # shape syntax and may silently upgrade the requested opset.
    if "dynamo" in export_parameters:
        kwargs["dynamo"] = False

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The feature will be removed\. Please remove usage of this function",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                spec.inference_module,
                spec.example_inputs,
                str(output_path),
                **kwargs,
            )
    except ModuleNotFoundError as exc:
        raise OnnxGraphExportError(
            "ONNX export dependencies are incomplete; install pyimgano[onnx-export]."
        ) from exc
    except Exception as exc:
        raise OnnxGraphExportError(f"Failed to export fitted ONNX graph: {exc}") from exc

    try:
        import onnx
    except ModuleNotFoundError as exc:  # pragma: no cover - capability gate
        raise OnnxGraphExportError(
            "ONNX structural verification requires pyimgano[onnx-export]."
        ) from exc
    try:
        model = onnx.load(str(output_path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise OnnxGraphExportError(f"Exported ONNX graph failed validation: {exc}") from exc

    graph_inputs = [str(item.name) for item in model.graph.input]
    graph_outputs = [str(item.name) for item in model.graph.output]
    if graph_inputs != list(spec.input_names) or graph_outputs != list(spec.output_names):
        raise OnnxGraphExportError(
            "Exported ONNX I/O names do not match the graph contract: "
            f"inputs={graph_inputs!r}, outputs={graph_outputs!r}."
        )
    external = []
    for initializer in model.graph.initializer:
        if int(initializer.data_location) == int(onnx.TensorProto.EXTERNAL):
            external.append(str(initializer.name))
    if external:
        raise OnnxGraphExportError(
            "Certified ae_resnet_unet export must be a self-contained ONNX file; "
            f"external initializers found: {external!r}."
        )
    opsets = [
        int(item.version)
        for item in model.opset_import
        if str(getattr(item, "domain", "")) in {"", "ai.onnx"}
    ]
    if not opsets or int(opset) not in opsets:
        raise OnnxGraphExportError(
            f"Exported ONNX graph did not preserve requested opset {opset}: {opsets!r}."
        )
    return OnnxGraphInfo(
        path=output_path,
        opset=int(opset),
        ir_version=int(model.ir_version),
    )


__all__ = [
    "DEFAULT_ONNX_OPSET",
    "OnnxGraphExportError",
    "OnnxGraphInfo",
    "export_onnx_graph",
]
