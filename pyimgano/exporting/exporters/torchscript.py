from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyimgano.exporting.types import GraphExportSpec


class TorchScriptGraphExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class TorchScriptGraphInfo:
    path: Path


def export_torchscript_graph(
    spec: GraphExportSpec,
    path: str | Path,
) -> TorchScriptGraphInfo:
    """Trace, freeze, save, and reload one full-detector TorchScript graph."""

    if len(spec.example_inputs) != 1 or len(spec.input_names) != 1:
        raise TorchScriptGraphExportError(
            "TorchScript single-graph export requires exactly one input."
        )
    if tuple(spec.output_names) != ("score", "anomaly_map"):
        raise TorchScriptGraphExportError(
            "Certified autoencoder TorchScript export requires score and anomaly_map outputs."
        )
    try:
        import torch

        from pyimgano.utils.torchscript_safe import freeze_module, load_module, trace_module
    except ModuleNotFoundError as exc:  # pragma: no cover - capability gate
        raise TorchScriptGraphExportError("TorchScript export requires pyimgano[torch].") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        module = spec.inference_module.to("cpu").eval()
        inputs = tuple(value.to("cpu") for value in spec.example_inputs)
        with torch.inference_mode():
            traced = trace_module(module, inputs)
            frozen = freeze_module(traced.eval())
        frozen.save(str(output_path))
        # A save-only check is insufficient: reload the exact published bytes.
        loaded = load_module(
            output_path,
            map_location=torch.device("cpu"),
            trusted=True,
        )
        with torch.inference_mode():
            outputs = loaded(*inputs)
    except Exception as exc:
        raise TorchScriptGraphExportError(
            f"Failed to export/reload fitted TorchScript graph: {exc}"
        ) from exc
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise TorchScriptGraphExportError(
            "Reloaded TorchScript graph did not return score and anomaly_map."
        )
    return TorchScriptGraphInfo(path=output_path)


__all__ = [
    "TorchScriptGraphExportError",
    "TorchScriptGraphInfo",
    "export_torchscript_graph",
]
