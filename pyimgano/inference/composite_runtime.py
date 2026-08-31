from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
from pyimgano.models.protocols import normalize_anomaly_maps, normalize_scores


def _as_feature_matrix(value: Any, *, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ArtifactRuntimeError("Embedding graph output must have a batch dimension.")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim == 2:
        pass
    elif array.ndim == 3:
        array = np.mean(array, axis=1)
    elif array.ndim == 4:
        array = np.mean(array, axis=(2, 3))
    else:
        array = array.reshape(int(array.shape[0]), -1)
    if array.ndim != 2 or int(array.shape[0]) != int(batch_size):
        raise ArtifactRuntimeError(
            "Embedding graph must return one feature row per image; "
            f"got shape {tuple(array.shape)} for batch {batch_size}."
        )
    features = np.asarray(array, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ArtifactRuntimeError("Embedding graph returned non-finite features.")
    return features


def _prepare_exact_embedding_batch(
    inputs: Sequence[Any], contract: Mapping[str, Any]
) -> np.ndarray:
    """Mirror the canonical ONNX/TorchScript extractor preprocessing exactly.

    In particular, the source extractors stack transposed CHW rows without making
    the result C-contiguous.  Preserving those strides matters: Torch reduction
    kernels can otherwise round ties differently, which is observable to ECOD's
    empirical ranks.
    """

    if (
        contract.get("dtype") != "float32"
        or contract.get("layout") != "NCHW"
        or contract.get("color_space") != "RGB"
    ):
        raise ArtifactRuntimeError("Unsupported composite embedding input contract.")
    size = contract.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ArtifactRuntimeError("Composite embedding size must be [height, width].")
    height, width = int(size[0]), int(size[1])
    resize = contract.get("resize")
    if not isinstance(resize, Mapping) or dict(resize) != {
        "mode": "stretch",
        "interpolation": "bilinear",
    }:
        raise ArtifactRuntimeError("Unsupported composite embedding resize contract.")
    scale = contract.get("scale")
    if not isinstance(scale, Mapping) or dict(scale) != {"divisor": 255.0}:
        raise ArtifactRuntimeError("Unsupported composite embedding scale contract.")
    normalize = contract.get("normalize")
    if not isinstance(normalize, Mapping):
        raise ArtifactRuntimeError("Composite embedding normalization is required.")
    try:
        mean = np.asarray(normalize["mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(normalize["std"], dtype=np.float32).reshape(3, 1, 1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactRuntimeError(
            "Composite embedding normalization must contain RGB mean/std triplets."
        ) from exc
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std == 0):
        raise ArtifactRuntimeError("Composite embedding normalization is invalid.")

    from PIL import Image

    rows: list[np.ndarray] = []
    for item in inputs:
        if isinstance(item, (str, Path)):
            with Image.open(str(item)) as source:
                image = source.convert("RGB")
        elif isinstance(item, np.ndarray):
            array = np.asarray(item)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ArtifactRuntimeError(
                    "Composite runtime arrays must be canonical RGB uint8/HWC."
                )
            image = Image.fromarray(array, mode="RGB")
        else:
            raise ArtifactRuntimeError(
                "Composite runtime inputs must be image paths or RGB uint8 arrays."
            )
        image = image.resize((width, height), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
        chw = np.transpose(array, (2, 0, 1)).astype(np.float32) / 255.0
        chw = (np.asarray(chw, dtype=np.float32) - mean) / std
        rows.append(np.asarray(chw, dtype=np.float32))
    return np.stack(rows, axis=0)


class OnnxEmbeddingComponentRuntime:
    """Manifest-driven ONNX embedding node for a composite detector."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_contract: Mapping[str, Any],
        output_contract: Mapping[str, Any],
        batch_size: int,
        allowed_providers: Any,
        verified_providers: Any,
        providers: Any = None,
        device: str | None = None,
        session_options: Mapping[str, Any] | None = None,
        ort_module: Any | None = None,
        session: Any | None = None,
    ) -> None:
        from pyimgano.inference.onnx_runtime import (
            _provider_names,
            build_onnx_session_options,
            resolve_onnx_providers,
        )

        self.model_path = Path(model_path)
        self.input_contract = dict(input_contract)
        self.output_contract = dict(output_contract)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ArtifactRuntimeError("Composite embedding batch_size must be positive.")
        if self.output_contract.get("reduction") != "auto_2d_v1":
            raise ArtifactRuntimeError("Unsupported embedding reduction contract.")
        if session is None:
            if ort_module is None:
                from pyimgano.utils.optional_deps import require

                ort_module = require(
                    "onnxruntime",
                    extra="onnx-runtime",
                    purpose="loading a composite ONNX embedding component",
                )
            specs, provider_warnings = resolve_onnx_providers(
                ort_module,
                allowed=allowed_providers,
                verified=verified_providers,
                providers=providers,
                device=device,
            )
            names = _provider_names(specs)
            options = [dict(item["options"]) for item in specs]
            kwargs: dict[str, Any] = {
                "providers": names,
                "provider_options": options,
            }
            built_options = build_onnx_session_options(ort_module, session_options)
            if built_options is not None:
                kwargs["sess_options"] = built_options
            session = ort_module.InferenceSession(str(self.model_path), **kwargs)
        else:
            provider_warnings = []
        self.session = session
        inputs = list(session.get_inputs())
        if len(inputs) != 1:
            raise ArtifactRuntimeError(
                f"Composite ONNX embedding must expose exactly one input; got {len(inputs)}."
            )
        input_name = str(self.input_contract.get("name", "")).strip()
        if not input_name or str(inputs[0].name) != input_name:
            raise ArtifactRuntimeError(
                f"Composite ONNX input mismatch: manifest={input_name!r}, "
                f"graph={getattr(inputs[0], 'name', None)!r}."
            )
        outputs = {str(item.name) for item in session.get_outputs()}
        output_name = str(self.output_contract.get("name", "")).strip()
        if not output_name or output_name not in outputs:
            raise ArtifactRuntimeError(
                f"Composite ONNX embedding output {output_name!r} is absent; "
                f"graph outputs={sorted(outputs)}."
            )
        selected: list[str] = []
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            selected = [str(item) for item in get_providers()]
        self.runtime_info = {
            "backend": "onnxruntime",
            "providers": selected,
            "selected_provider": selected[0] if selected else None,
            "warnings": list(provider_warnings),
        }

    def extract(self, inputs: Sequence[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 1), dtype=np.float64)
        input_name = str(self.input_contract["name"])
        output_name = str(self.output_contract["name"])
        rows: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch_items = items[start : start + self.batch_size]
            batch = _prepare_exact_embedding_batch(batch_items, self.input_contract)
            values = self.session.run([output_name], {input_name: batch})
            if len(values) != 1:
                raise ArtifactRuntimeError("ONNX embedding runtime returned no selected output.")
            rows.append(_as_feature_matrix(values[0], batch_size=len(batch_items)))
        result = np.concatenate(rows, axis=0)
        if int(result.shape[0]) != len(items):  # pragma: no cover - per-batch invariant
            raise ArtifactRuntimeError("ONNX embedding row count changed during batching.")
        return np.asarray(result, dtype=np.float64)


def _select_torchscript_embedding(output: Any, contract: Mapping[str, Any]) -> Any:
    if hasattr(output, "detach"):
        return output
    if isinstance(output, Mapping):
        key = contract.get("output_key")
        if not isinstance(key, str) or key not in output:
            raise ArtifactRuntimeError(
                "TorchScript mapping output requires its declared output_key."
            )
        return output[key]
    if isinstance(output, (tuple, list)):
        index = int(contract.get("output_index", 0))
        if index < 0 or index >= len(output):
            raise ArtifactRuntimeError(
                f"TorchScript embedding output index {index} is outside tuple length {len(output)}."
            )
        return output[index]
    raise ArtifactRuntimeError(
        f"Unsupported TorchScript embedding output type: {type(output).__name__}."
    )


class TorchScriptEmbeddingComponentRuntime:
    """Manifest-driven TorchScript embedding node for a composite detector."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_contract: Mapping[str, Any],
        output_contract: Mapping[str, Any],
        batch_size: int,
        device: str = "cpu",
        trust_checkpoint: bool = False,
        torch_module: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_contract = dict(input_contract)
        self.output_contract = dict(output_contract)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ArtifactRuntimeError("Composite embedding batch_size must be positive.")
        if self.output_contract.get("reduction") != "auto_2d_v1":
            raise ArtifactRuntimeError("Unsupported embedding reduction contract.")
        if torch_module is None:
            from pyimgano.utils.optional_deps import require

            torch_module = require(
                "torch",
                extra="torch",
                purpose="loading a composite TorchScript embedding component",
            )
        self.torch = torch_module
        device_name = str(device).strip().lower()
        if device_name.startswith("cuda") and not bool(torch_module.cuda.is_available()):
            raise ArtifactRuntimeError(
                f"TorchScript device {device_name!r} requested but CUDA is unavailable."
            )
        self.device = torch_module.device(device_name)
        supplied_model = model is not None
        if model is None:
            if not bool(trust_checkpoint):
                raise ArtifactRuntimeError(
                    "TorchScript component deserialization requires explicit trust."
                )
            from pyimgano.utils.torchscript_safe import load_module

            model = load_module(
                self.model_path,
                map_location=self.device,
                trusted=True,
            )
        # load_module(map_location=...) already places serialized parameters.  A
        # redundant ScriptModule.to("cpu") can select a different JIT execution
        # plan and perturb reductions enough to change empirical ECOD ranks.
        self.model = (
            model.to(self.device)
            if supplied_model and callable(getattr(model, "to", None))
            else model
        )
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()
        self.runtime_info = {
            "backend": "torchscript",
            "device": str(self.device),
            "providers": [],
            "selected_provider": None,
        }

    def extract(self, inputs: Sequence[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 1), dtype=np.float64)
        rows: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch_items = items[start : start + self.batch_size]
            batch = _prepare_exact_embedding_batch(batch_items, self.input_contract)
            tensor = self.torch.from_numpy(batch).to(self.device)
            # Match TorchscriptEmbedExtractor exactly.  Some reduction kernels
            # choose observably different floating-point paths under
            # inference_mode(), which can change empirical ECOD ranks at ties.
            with self.torch.no_grad():
                output = self.model(tensor)
            selected = _select_torchscript_embedding(output, self.output_contract)
            if not hasattr(selected, "detach"):
                raise ArtifactRuntimeError("Selected TorchScript embedding output is not a tensor.")
            value = selected.detach().to("cpu").numpy()
            rows.append(_as_feature_matrix(value, batch_size=len(batch_items)))
        result = np.concatenate(rows, axis=0)
        if int(result.shape[0]) != len(items):  # pragma: no cover - per-batch invariant
            raise ArtifactRuntimeError("TorchScript embedding row count changed during batching.")
        return np.asarray(result, dtype=np.float64)


class CompositeArtifactRuntime:
    """Compose a verified graph/component runtime with fitted detector state.

    The preferred integration is an explicit registered adapter callable. The
    feature-runtime/fitted-core fallback exists for simple classical cores and is
    intentionally small: it never imports a class named by artifact data.
    """

    def __init__(
        self,
        *,
        component_runtime: Any,
        fitted_core: Any,
        adapter: Callable[..., Any] | None = None,
        adapter_id: str | None = None,
        runtime_info: Mapping[str, Any] | None = None,
    ) -> None:
        self.component_runtime = component_runtime
        self.fitted_core = fitted_core
        self.adapter = adapter
        self.adapter_id = adapter_id
        info = dict(runtime_info or {})
        info.setdefault("backend", "composite")
        info.setdefault("adapter_id", adapter_id)
        child_info = getattr(component_runtime, "runtime_info", None)
        if isinstance(child_info, Mapping):
            info.setdefault("component_runtime", dict(child_info))
            info.setdefault("providers", list(child_info.get("providers", [])))
            info.setdefault("selected_provider", child_info.get("selected_provider"))
        self.runtime_info = info

    def _features(self, inputs: Sequence[Any]) -> Any:
        extractor = getattr(self.component_runtime, "extract", None)
        if callable(extractor):
            return extractor(inputs)
        transformer = getattr(self.component_runtime, "transform", None)
        if callable(transformer):
            return transformer(inputs)
        decision = getattr(self.component_runtime, "decision_function", None)
        if callable(decision):
            return decision(inputs)
        if callable(self.component_runtime):
            return self.component_runtime(inputs)
        raise ArtifactRuntimeError("Composite component runtime exposes no feature operation.")

    def score_and_maps(
        self, inputs: Sequence[Any], *, include_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        items = list(inputs)
        if not items:
            return np.zeros((0,), dtype=np.float32), None
        if self.adapter is not None:
            result = self.adapter(
                component_runtime=self.component_runtime,
                fitted_core=self.fitted_core,
                inputs=items,
                include_maps=bool(include_maps),
            )
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ArtifactRuntimeError("Composite adapter must return (scores, maps).")
            scores, maps = result
        else:
            features = self._features(items)
            scorer = getattr(self.fitted_core, "decision_function", None)
            if not callable(scorer):
                scorer = getattr(self.fitted_core, "score_samples", None)
            if not callable(scorer):
                raise ArtifactRuntimeError("Composite fitted core exposes no scoring operation.")
            scores = scorer(features)
            maps = None
            if include_maps:
                map_fn = getattr(self.fitted_core, "predict_anomaly_map", None)
                if callable(map_fn):
                    maps = map_fn(features)

        normalized_scores = normalize_scores(scores, n_expected=len(items))
        if maps is None or not include_maps:
            return normalized_scores, None
        return normalized_scores, normalize_anomaly_maps(maps, n_expected=len(items))

    def decision_function(self, inputs: Sequence[Any]) -> np.ndarray:
        return self.score_and_maps(inputs, include_maps=False)[0]

    def predict_anomaly_map(self, inputs: Sequence[Any]) -> np.ndarray:
        _scores, maps = self.score_and_maps(inputs, include_maps=True)
        if maps is None:
            raise ArtifactRuntimeError("Composite artifact has no anomaly-map capability.")
        return maps


__all__ = [
    "CompositeArtifactRuntime",
    "OnnxEmbeddingComponentRuntime",
    "TorchScriptEmbeddingComponentRuntime",
]
