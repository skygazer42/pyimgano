from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.exporting.types import (
    ArtifactFormat,
    CapabilityAvailability,
    CheckpointContract,
    ExportCapability,
    ExportLayout,
    ExportStatus,
    GraphExportSpec,
    NativeExportContext,
    ProbeSpec,
    SerializationKind,
)

AUTOENCODER_ADAPTER_ID = "pyimgano.ae-resnet-unet"
AUTOENCODER_ADAPTER_VERSION = 1
AUTOENCODER_CODEC_ID = "pyimgano.torch-state-dict"
AUTOENCODER_CODEC_VERSION = 1
AUTOENCODER_STATE_SCHEMA_VERSION = 1

_MODEL_NAME = "ae_resnet_unet"
_MODEL_STATE_KEYS = frozenset(
    {
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "encoder.4.weight",
        "encoder.4.bias",
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
        "decoder.4.weight",
        "decoder.4.bias",
    }
)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class AutoencoderAdapterError(RuntimeError):
    pass


def _array_view(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        normalized = detach()
        cpu = getattr(normalized, "cpu", None)
        if callable(cpu):
            normalized = cpu()
        numpy_fn = getattr(normalized, "numpy", None)
        if callable(numpy_fn):
            return np.asarray(numpy_fn())
    return None


def _normalized_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoencoderAdapterError("Autoencoder fitted state is missing its architecture.")
    unknown = sorted(set(value) - {"image_size", "latent_channels", "base_channels"})
    if unknown:
        raise AutoencoderAdapterError(
            f"Autoencoder fitted-state architecture has unknown fields: {unknown!r}."
        )
    image_size = value.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise AutoencoderAdapterError("Autoencoder image_size must contain [height, width].")
    height, width = int(image_size[0]), int(image_size[1])
    latent = int(value.get("latent_channels", 0))
    base = int(value.get("base_channels", 0))
    if height <= 0 or width <= 0 or latent <= 0 or base <= 0:
        raise AutoencoderAdapterError("Autoencoder architecture dimensions must be positive.")
    if height % 8 or width % 8:
        raise AutoencoderAdapterError(
            "ae_resnet_unet export requires image_size dimensions divisible by 8."
        )
    return {
        "image_size": [height, width],
        "latent_channels": latent,
        "base_channels": base,
    }


def _detector_config(detector: Any) -> dict[str, Any]:
    cfg = getattr(detector, "cfg", None)
    if cfg is None:
        raise AutoencoderAdapterError("Detector does not expose the ae_resnet_unet config.")
    return _normalized_config(
        {
            "image_size": getattr(cfg, "image_size", None),
            "latent_channels": getattr(cfg, "latent_channels", None),
            "base_channels": getattr(cfg, "base_channels", None),
        }
    )


def _validate_state_shapes(state: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    c = int(config["base_channels"])
    z = int(config["latent_channels"])
    expected = {
        "encoder.0.weight": (c, 3, 3, 3),
        "encoder.0.bias": (c,),
        "encoder.2.weight": (c * 2, c, 3, 3),
        "encoder.2.bias": (c * 2,),
        "encoder.4.weight": (z, c * 2, 3, 3),
        "encoder.4.bias": (z,),
        "decoder.0.weight": (z, c * 2, 4, 4),
        "decoder.0.bias": (c * 2,),
        "decoder.2.weight": (c * 2, c, 4, 4),
        "decoder.2.bias": (c,),
        "decoder.4.weight": (c, 3, 4, 4),
        "decoder.4.bias": (3,),
    }
    keys = {str(key) for key in state}
    if keys != _MODEL_STATE_KEYS:
        missing = sorted(_MODEL_STATE_KEYS - keys)
        extra = sorted(keys - _MODEL_STATE_KEYS)
        raise AutoencoderAdapterError(
            "Autoencoder fitted state does not match adapter schema v1: "
            f"missing={missing!r}, extra={extra!r}."
        )
    total_bytes = 0
    for name, shape in expected.items():
        array = _array_view(state[name])
        if array is None:
            raise AutoencoderAdapterError(f"Autoencoder weight {name!r} is not a tensor/array.")
        if tuple(int(item) for item in array.shape) != shape:
            raise AutoencoderAdapterError(
                f"Autoencoder weight {name!r} has shape {tuple(array.shape)!r}; "
                f"expected {shape!r}."
            )
        if str(array.dtype) != "float32":
            raise AutoencoderAdapterError(
                f"Autoencoder weight {name!r} has dtype {array.dtype!s}; expected float32."
            )
        if not np.isfinite(array).all():
            raise AutoencoderAdapterError(
                f"Autoencoder weight {name!r} contains non-finite values."
            )
        total_bytes += int(array.nbytes)
    if total_bytes > 512 * 1024 * 1024:
        raise AutoencoderAdapterError("Autoencoder fitted state exceeds the codec byte limit.")


class AutoencoderStateCodec:
    """Safe, allowlisted tensor codec for the canonical convolutional autoencoder."""

    codec_id = AUTOENCODER_CODEC_ID
    codec_version = AUTOENCODER_CODEC_VERSION
    state_schema_version = AUTOENCODER_STATE_SCHEMA_VERSION
    model_names = (_MODEL_NAME,)

    def encode(self, detector: Any) -> Mapping[str, Any]:
        model = getattr(detector, "model", None)
        state_dict_fn = getattr(model, "state_dict", None)
        if model is None or not callable(state_dict_fn):
            raise AutoencoderAdapterError(
                "ae_resnet_unet must be fitted before its state can be exported."
            )
        state = {
            "architecture": _detector_config(detector),
            "model_state_dict": {
                str(name): tensor.detach().cpu().clone() for name, tensor in state_dict_fn().items()
            },
        }
        self.validate_state(state)
        return state

    def validate_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise AutoencoderAdapterError("Autoencoder fitted state must be a mapping.")
        keys = {str(key) for key in state}
        if keys != {"architecture", "model_state_dict"}:
            raise AutoencoderAdapterError(
                "Autoencoder fitted state must contain only architecture and model_state_dict."
            )
        config = _normalized_config(state.get("architecture"))
        weights = state.get("model_state_dict")
        if not isinstance(weights, Mapping):
            raise AutoencoderAdapterError("Autoencoder model_state_dict must be a mapping.")
        _validate_state_shapes(weights, config)

    def decode(self, detector: Any, state: Mapping[str, Any]) -> None:
        self.validate_state(state)
        encoded_config = _normalized_config(state["architecture"])
        actual_config = _detector_config(detector)
        if encoded_config != actual_config:
            raise AutoencoderAdapterError(
                "Autoencoder fitted-state architecture does not match constructor kwargs: "
                f"state={encoded_config!r}, detector={actual_config!r}."
            )

        model = getattr(detector, "model", None)
        if model is None:
            build_model = getattr(detector, "build_model", None)
            if not callable(build_model):
                raise AutoencoderAdapterError("Detector cannot rebuild its autoencoder model.")
            model = build_model()
        load_state_dict = getattr(model, "load_state_dict", None)
        if not callable(load_state_dict):
            raise AutoencoderAdapterError("Detector model cannot restore a state dictionary.")
        load_state_dict(dict(state["model_state_dict"]), strict=True)
        device = getattr(detector, "device", "cpu")
        move = getattr(model, "to", None)
        if callable(move):
            model = move(device)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        detector.model = model
        detector.is_fitted_ = True


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _requirements(format: ArtifactFormat) -> tuple[tuple[str, ...], str]:
    if format is ArtifactFormat.NATIVE:
        return ("torch", "torchvision"), "torch"
    if format is ArtifactFormat.TORCHSCRIPT:
        return ("torch", "torchvision"), "torch"
    if format is ArtifactFormat.ONNX:
        return ("torch", "torchvision", "onnx", "onnxruntime"), "onnx-export"
    return (
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
        "openvino",
    ), "openvino-export"


def _missing_requirements(format: ArtifactFormat) -> tuple[tuple[str, ...], str]:
    modules, extra = _requirements(format)
    return tuple(name for name in modules if not _module_available(name)), extra


def _context_mapping(context: NativeExportContext | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(context, Mapping):
        return context
    return {
        "model_name": context.model_name,
        "model_kwargs": context.model_kwargs,
        "checkpoint_contract": context.checkpoint_contract,
    }


def _checkpoint_from_context(
    context: NativeExportContext | Mapping[str, Any],
) -> CheckpointContract | None:
    if isinstance(context, NativeExportContext):
        return context.checkpoint_contract
    raw = context.get("checkpoint_contract")
    if isinstance(raw, CheckpointContract):
        return raw
    if isinstance(raw, Mapping):
        try:
            return CheckpointContract.from_mapping(raw)
        except (TypeError, ValueError):
            return None
    return None


def _configured_image_size(context: NativeExportContext | Mapping[str, Any]) -> tuple[int, int]:
    values = _context_mapping(context)
    kwargs = values.get("model_kwargs", {})
    kwargs = kwargs if isinstance(kwargs, Mapping) else {}
    raw = kwargs.get("image_size", 256)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    value = int(raw)
    return value, value


def _fixed_probe_images(image_size: Sequence[int]) -> tuple[np.ndarray, ...]:
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise AutoencoderAdapterError("Probe image dimensions must be positive.")
    yy = np.arange(height, dtype=np.uint16)[:, None]
    xx = np.arange(width, dtype=np.uint16)[None, :]
    gradient = np.empty((height, width, 3), dtype=np.uint8)
    gradient[..., 0] = ((xx * 255) // max(width - 1, 1)).astype(np.uint8)
    gradient[..., 1] = ((yy * 255) // max(height - 1, 1)).astype(np.uint8)
    gradient[..., 2] = ((xx + yy) % 256).astype(np.uint8)
    checker = (((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)
    checker_rgb = np.stack(
        [checker, np.roll(checker, 1, axis=0), np.roll(checker, 1, axis=1)],
        axis=-1,
    )
    return (
        np.zeros((height, width, 3), dtype=np.uint8),
        gradient,
        np.ascontiguousarray(checker_rgb),
    )


def _evaluate_detector(detector: Any, inputs: Sequence[Any]) -> dict[str, np.ndarray]:
    images = list(inputs)
    decision = getattr(detector, "decision_function", None)
    if not callable(decision):
        raise AutoencoderAdapterError("Detector does not expose decision_function().")
    scores = np.asarray(decision(images), dtype=np.float32).reshape(-1)
    if scores.shape != (len(images),):
        raise AutoencoderAdapterError(
            f"Detector probe returned invalid score shape {scores.shape!r}."
        )
    map_fn = getattr(detector, "get_anomaly_map", None)
    if not callable(map_fn):
        raise AutoencoderAdapterError("Detector does not expose get_anomaly_map().")
    maps = np.stack(
        [np.asarray(map_fn(image), dtype=np.float32) for image in images],
        axis=0,
    )
    return {"score": scores, "anomaly_map": maps}


def _parity_metrics(
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
        denominator = np.maximum(np.abs(expected), np.float32(1e-12))
        relative = absolute / denominator
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
    return {
        "passed": bool(passed),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "outputs": outputs,
    }


def _make_inference_module(model: Any) -> Any:
    import torch

    class _AutoencoderInferenceModule(torch.nn.Module):
        def __init__(self, autoencoder: Any) -> None:
            super().__init__()
            self.autoencoder = autoencoder

        def forward(self, value: Any) -> tuple[Any, Any]:
            reconstruction = self.autoencoder(value)
            residual = reconstruction - value
            score = residual.square().flatten(1).mean(dim=1)
            anomaly_map = residual.abs().mean(dim=1)
            return score, anomaly_map

    return _AutoencoderInferenceModule(model)


class AutoencoderExportAdapter:
    """Certified fitted export adapter for the canonical ``ae_resnet_unet`` model."""

    adapter_id = AUTOENCODER_ADAPTER_ID
    adapter_version = AUTOENCODER_ADAPTER_VERSION
    model_names = (_MODEL_NAME,)
    state_codec_id = AUTOENCODER_CODEC_ID
    state_codec_version = AUTOENCODER_CODEC_VERSION
    state_schema_version = AUTOENCODER_STATE_SCHEMA_VERSION
    native_runtime_versions = {"torch": ">=1.9"}

    def inspect_source(self, source: Any) -> Mapping[str, Any]:
        model = getattr(source, "model", None)
        state_dict = getattr(model, "state_dict", None)
        payload: dict[str, Any] = {
            "model_name": _MODEL_NAME,
            "fitted": bool(model is not None and callable(state_dict)),
            "has_graph_state": bool(model is not None and callable(state_dict)),
        }
        if getattr(source, "cfg", None) is not None:
            payload["architecture"] = _detector_config(source)
        return payload

    def declared_capability(self, format: ArtifactFormat) -> ExportCapability:
        if format is ArtifactFormat.NATIVE:
            return ExportCapability(
                format=format,
                status=ExportStatus.SUPPORTED,
                layout=ExportLayout.NATIVE_DETECTOR,
                conditions=("complete_verified_checkpoint", "torch_runtime"),
            )
        if format in {
            ArtifactFormat.ONNX,
            ArtifactFormat.TORCHSCRIPT,
            ArtifactFormat.OPENVINO,
        }:
            condition = {
                ArtifactFormat.ONNX: "onnx_export_and_runtime_extras",
                ArtifactFormat.TORCHSCRIPT: "torch_runtime",
                ArtifactFormat.OPENVINO: "verified_onnx_then_openvino_conversion",
            }[format]
            return ExportCapability(
                format=format,
                status=ExportStatus.CONDITIONAL,
                layout=ExportLayout.SINGLE_GRAPH,
                conditions=("complete_verified_checkpoint", condition),
                reason_code="requires_concrete_export_context",
                remediation=(
                    "Provide a complete certified checkpoint and install the declared "
                    "runtime/export extras."
                ),
            )
        return ExportCapability.unsupported(format, reason_code="format_not_certified")

    def effective_capability(
        self,
        format: ArtifactFormat,
        *,
        context: Mapping[str, Any] | NativeExportContext,
    ) -> ExportCapability:
        declared = self.declared_capability(format)
        missing, extra = _missing_requirements(format)
        if missing:
            return ExportCapability.unsupported(
                format,
                reason_code="missing_export_dependency",
                remediation=(
                    f"Install pyimgano[{extra}] before requesting {format.value} export; "
                    f"missing modules: {', '.join(missing)}."
                ),
                availability=CapabilityAvailability.MISSING_EXTRA,
                conditions=declared.conditions,
            )

        try:
            height, width = _configured_image_size(context)
        except (TypeError, ValueError):
            return ExportCapability.unsupported(
                format,
                reason_code="invalid_model_config",
                remediation="Set image_size to an integer or [height, width].",
            )
        if height <= 0 or width <= 0 or height % 8 or width % 8:
            return ExportCapability.unsupported(
                format,
                reason_code="unsupported_image_size",
                remediation=(
                    "ae_resnet_unet export requires positive image_size dimensions "
                    "divisible by 8."
                ),
            )

        values = _context_mapping(context)
        if str(values.get("phase", "")).strip() == "pre_training":
            return declared

        contract = _checkpoint_from_context(context)
        if contract is None or not contract.strict_exportable:
            return ExportCapability.unsupported(
                format,
                reason_code="checkpoint_incomplete",
                remediation=(
                    "Train and persist the model through the certified adapter round-trip "
                    "before exporting it."
                ),
                conditions=declared.conditions,
            )
        mismatch = self._checkpoint_mismatch(contract)
        if mismatch is not None:
            return ExportCapability.unsupported(
                format,
                reason_code=mismatch,
                remediation="Recreate the checkpoint with the registered ae_resnet_unet adapter.",
                conditions=declared.conditions,
            )
        return ExportCapability(
            format=format,
            status=ExportStatus.SUPPORTED,
            layout=(
                ExportLayout.NATIVE_DETECTOR
                if format is ArtifactFormat.NATIVE
                else ExportLayout.SINGLE_GRAPH
            ),
            conditions=declared.conditions,
        )

    def _checkpoint_mismatch(self, contract: CheckpointContract) -> str | None:
        if (
            contract.adapter_id != self.adapter_id
            or int(contract.adapter_version or 0) != self.adapter_version
        ):
            return "checkpoint_adapter_mismatch"
        if (
            contract.codec_id != self.state_codec_id
            or int(contract.codec_version or 0) != self.state_codec_version
            or int(contract.state_schema_version or 0) != self.state_schema_version
        ):
            return "checkpoint_codec_mismatch"
        if contract.serialization is not SerializationKind.SAFE_DATA or bool(
            contract.requires_trust
        ):
            return "checkpoint_requires_trust"
        return None

    def validate_checkpoint_contract(
        self,
        contract: CheckpointContract,
        *,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> None:
        del context
        if not contract.strict_exportable:
            raise AutoencoderAdapterError(
                "ae_resnet_unet export requires a complete checkpoint with explicit "
                "round-trip verification."
            )
        mismatch = self._checkpoint_mismatch(contract)
        if mismatch is not None:
            raise AutoencoderAdapterError(
                f"ae_resnet_unet checkpoint contract is not compatible: {mismatch}."
            )

    def restore_state(self, detector: Any, checkpoint: str | Path) -> None:
        from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

        load_checkpoint_into_detector(detector, checkpoint, trusted=False)
        model = getattr(detector, "model", None)
        if model is None:
            raise AutoencoderAdapterError("Checkpoint restore did not produce detector.model.")
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        detector.is_fitted_ = True

    def build_probe_spec(
        self,
        detector: Any,
        *,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> ProbeSpec:
        del context
        config = _detector_config(detector)
        return ProbeSpec(
            # Contract: each item is a canonical raw RGB uint8/HWC image and the
            # tuple can be passed directly (after list()) to decision_function().
            inputs=_fixed_probe_images(config["image_size"]),
            expected_outputs=("score", "anomaly_map"),
            absolute_tolerance=1e-5,
            relative_tolerance=1e-4,
        )

    def evaluate_probe(
        self,
        detector: Any,
        spec: ProbeSpec | None = None,
    ) -> Mapping[str, np.ndarray]:
        selected = spec if spec is not None else self.build_probe_spec(detector)
        return _evaluate_detector(detector, selected.inputs)

    def verify_roundtrip(
        self,
        original: Any,
        restored: Any,
        spec: ProbeSpec | None = None,
    ) -> Mapping[str, Any]:
        selected = spec if spec is not None else self.build_probe_spec(original)
        report = _parity_metrics(
            self.evaluate_probe(original, selected),
            self.evaluate_probe(restored, selected),
            absolute_tolerance=selected.absolute_tolerance,
            relative_tolerance=selected.relative_tolerance,
        )
        if not bool(report["passed"]):
            raise AutoencoderAdapterError(
                f"ae_resnet_unet checkpoint round-trip parity failed: {report!r}."
            )
        return report

    def build_graph_export_spec(
        self,
        detector: Any,
        *,
        format: ArtifactFormat,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> GraphExportSpec:
        del context
        if format not in {
            ArtifactFormat.ONNX,
            ArtifactFormat.TORCHSCRIPT,
            ArtifactFormat.OPENVINO,
        }:
            raise AutoencoderAdapterError(f"{format.value} is not a graph export format.")
        source_model = getattr(detector, "model", None)
        if source_model is None or not callable(getattr(source_model, "state_dict", None)):
            raise AutoencoderAdapterError(
                "ae_resnet_unet must be fitted/restored before graph export."
            )
        config = _detector_config(detector)
        model = copy.deepcopy(source_model).to("cpu").eval()
        module = _make_inference_module(model).to("cpu").eval()
        import torch

        height, width = config["image_size"]
        example = torch.zeros((1, 3, height, width), dtype=torch.float32)
        return GraphExportSpec(
            inference_module=module,
            example_inputs=(example,),
            input_names=("input",),
            output_names=("score", "anomaly_map"),
            dynamic_axes={
                "input": {0: "batch"},
                "score": {0: "batch"},
                "anomaly_map": {0: "batch"},
            },
            output_semantics={
                "score": "raw_reconstruction_mse",
                "anomaly_map": "raw_channel_mean_absolute_reconstruction_error",
                "operating_threshold_embedded": False,
            },
        )

    def build_input_contract(self, detector: Any) -> Mapping[str, Any]:
        height, width = _detector_config(detector)["image_size"]
        return {
            "kind": "image_batch",
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [height, width],
            "dynamic_axes": {"batch": True},
            "resize": {"mode": "stretch", "interpolation": "bilinear"},
            "scale": {"divisor": 255.0},
            "normalize": {"mean": list(_MEAN), "std": list(_STD)},
        }

    def build_output_contract(self) -> Mapping[str, Any]:
        return {
            "score": {
                "name": "score",
                "output_index": 0,
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            },
            "anomaly_map": {
                "name": "anomaly_map",
                "output_index": 1,
                "layout": "NHW",
                "transform": "identity",
                "resize_to_source": True,
            },
        }

    def build_runtime_spec(
        self,
        *,
        format: ArtifactFormat,
        context: NativeExportContext | Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        if format is ArtifactFormat.NATIVE:
            return {
                "backend": "pyimgano",
                "allowed_providers": [{"name": "CPU", "options": {}}],
                "verified_providers": [{"name": "CPU", "options": {}}],
            }
        if format is ArtifactFormat.ONNX:
            backend, provider = "onnxruntime", "CPUExecutionProvider"
        elif format is ArtifactFormat.TORCHSCRIPT:
            backend, provider = "torchscript", "CPU"
        elif format is ArtifactFormat.OPENVINO:
            backend, provider = "openvino", "CPU"
        else:
            raise AutoencoderAdapterError(f"Unsupported runtime format: {format.value}.")
        return {
            "backend": backend,
            "allowed_providers": [{"name": provider, "options": {}}],
            "verified_providers": [{"name": provider, "options": {}}],
        }

    def export_artifact(
        self,
        detector: Any,
        *,
        format: ArtifactFormat,
        context: NativeExportContext,
        out: str | Path,
        overwrite: bool = False,
    ) -> Any:
        if format is ArtifactFormat.NATIVE:
            from pyimgano.exporting.exporters.native import export_native

            return export_native(
                detector,
                context=context,
                out=out,
                adapter=self,
                overwrite=overwrite,
            )
        from pyimgano.exporting.exporters.single_graph import export_single_graph

        return export_single_graph(
            detector,
            format=format,
            context=context,
            out=out,
            adapter=self,
            overwrite=overwrite,
        )


AUTOENCODER_STATE_CODEC = AutoencoderStateCodec()
AUTOENCODER_EXPORT_ADAPTER = AutoencoderExportAdapter()


__all__ = [
    "AUTOENCODER_ADAPTER_ID",
    "AUTOENCODER_ADAPTER_VERSION",
    "AUTOENCODER_CODEC_ID",
    "AUTOENCODER_CODEC_VERSION",
    "AUTOENCODER_EXPORT_ADAPTER",
    "AUTOENCODER_STATE_CODEC",
    "AUTOENCODER_STATE_SCHEMA_VERSION",
    "AutoencoderAdapterError",
    "AutoencoderExportAdapter",
    "AutoencoderStateCodec",
]
