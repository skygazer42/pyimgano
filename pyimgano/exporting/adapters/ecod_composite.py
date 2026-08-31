from __future__ import annotations

import hashlib
import importlib.util
import math
import tempfile
from dataclasses import dataclass
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
    NativeExportContext,
    ProbeSpec,
    SerializationKind,
)

ECOD_COMPOSITE_ADAPTER_ID = "pyimgano.embedding-core-ecod"
ECOD_COMPOSITE_ADAPTER_VERSION = 1
ECOD_CORE_CODEC_ID = "pyimgano.core-ecod"
ECOD_CORE_CODEC_VERSION = 1
ECOD_CORE_STATE_SCHEMA_VERSION = 1

_ONNX_MODEL = "vision_onnx_ecod"
_TORCHSCRIPT_MODEL = "vision_torchscript_ecod"
_MODEL_NAMES = (_ONNX_MODEL, _TORCHSCRIPT_MODEL)
_MEGABYTE = 1024 * 1024
_MAX_STATE_BYTES = 512 * _MEGABYTE


class ECODCompositeAdapterError(RuntimeError):
    pass


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


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


def _model_name_from_context(context: NativeExportContext | Mapping[str, Any]) -> str:
    return str(_context_mapping(context).get("model_name", "")).strip()


def _expected_format(model_name: str) -> ArtifactFormat:
    if model_name == _ONNX_MODEL:
        return ArtifactFormat.ONNX
    if model_name == _TORCHSCRIPT_MODEL:
        return ArtifactFormat.TORCHSCRIPT
    raise ECODCompositeAdapterError(
        f"The ECOD composite adapter is not bound to model {model_name!r}."
    )


def _core_chain(detector: Any) -> tuple[Any, tuple[Any, ...]]:
    from pyimgano.models.ecod import CoreECOD

    if isinstance(detector, CoreECOD):
        return detector, (detector,)

    chain: list[Any] = [detector]
    candidate = getattr(detector, "detector", None)
    if candidate is not None:
        chain.append(candidate)
    nested = getattr(candidate, "detector", None)
    if nested is not None:
        chain.append(nested)
    if not isinstance(nested, CoreECOD):
        raise ECODCompositeAdapterError(
            "Expected a fitted VisionEmbeddingCoreDetector backed by core_ecod."
        )
    return nested, tuple(chain)


def _training_scores(chain: Sequence[Any]) -> np.ndarray:
    for value in chain:
        scores = getattr(value, "decision_scores_", None)
        if scores is None:
            continue
        normalized = np.asarray(scores, dtype=np.float64).reshape(-1)
        if normalized.size:
            return np.array(normalized, dtype=np.float64, copy=True)
    raise ECODCompositeAdapterError(
        "Fitted ECOD state is missing its training-score calibration array."
    )


def _require_float64_array(
    value: Any,
    *,
    field: str,
    rank: int,
    maximum_bytes: int = _MAX_STATE_BYTES,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ECODCompositeAdapterError(f"{field} must be a NumPy array.")
    array = np.asarray(value)
    if str(array.dtype) != "float64":
        raise ECODCompositeAdapterError(f"{field} must use float64, got {array.dtype!s}.")
    if int(array.ndim) != int(rank):
        raise ECODCompositeAdapterError(f"{field} must have rank {rank}, got {array.ndim}.")
    if int(array.nbytes) > int(maximum_bytes):
        raise ECODCompositeAdapterError(f"{field} exceeds the safe codec byte limit.")
    if not np.isfinite(array).all():
        raise ECODCompositeAdapterError(f"{field} contains non-finite values.")
    return array


class CoreECODStateCodec:
    """Complete non-executable fitted state for ECOD image-score computation.

    Operational thresholds and fitted labels deliberately do not appear in this
    schema. ``training_scores`` is retained as the learned score-calibration
    reference used by probability/confidence helpers, while raw scoring depends
    only on the empirical feature distribution and skew signs.
    """

    codec_id = ECOD_CORE_CODEC_ID
    codec_version = ECOD_CORE_CODEC_VERSION
    state_schema_version = ECOD_CORE_STATE_SCHEMA_VERSION
    model_names = _MODEL_NAMES

    def encode(self, detector: Any) -> Mapping[str, Any]:
        core, chain = _core_chain(detector)
        x_sorted = getattr(core, "_x_sorted", None)
        skew_sign = getattr(core, "_skew_sign", None)
        if x_sorted is None or skew_sign is None:
            raise ECODCompositeAdapterError(
                "core_ecod must be fitted before its state can be exported."
            )
        x_sorted_array = np.asarray(x_sorted, dtype=np.float64)
        skew_sign_array = np.asarray(skew_sign, dtype=np.float64)
        state = {
            "core_state": {
                "x_sorted": np.array(x_sorted_array, dtype=np.float64, copy=True),
                "skew_sign": np.array(skew_sign_array, dtype=np.float64, copy=True),
                "eps": float(getattr(core, "eps", 1e-12)),
            },
            "score_calibration": {
                "kind": "empirical_training_scores_v1",
                "training_scores": _training_scores(chain),
            },
            "feature_metadata": {
                "feature_dimension": int(x_sorted_array.shape[1]),
                "training_sample_count": int(x_sorted_array.shape[0]),
                "embedding_reduction": "auto_2d_v1",
            },
        }
        self.validate_state(state)
        return state

    def validate_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ECODCompositeAdapterError("ECOD fitted state must be a mapping.")
        if set(state) != {"core_state", "score_calibration", "feature_metadata"}:
            raise ECODCompositeAdapterError(
                "ECOD fitted state must contain only core_state, score_calibration, "
                "and feature_metadata."
            )
        core_state = state.get("core_state")
        calibration = state.get("score_calibration")
        metadata = state.get("feature_metadata")
        if not isinstance(core_state, Mapping) or set(core_state) != {
            "x_sorted",
            "skew_sign",
            "eps",
        }:
            raise ECODCompositeAdapterError("ECOD core_state does not match schema v1.")
        if not isinstance(calibration, Mapping) or set(calibration) != {
            "kind",
            "training_scores",
        }:
            raise ECODCompositeAdapterError("ECOD score_calibration does not match schema v1.")
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "feature_dimension",
            "training_sample_count",
            "embedding_reduction",
        }:
            raise ECODCompositeAdapterError("ECOD feature_metadata does not match schema v1.")

        x_sorted = _require_float64_array(
            core_state["x_sorted"], field="core_state.x_sorted", rank=2
        )
        skew_sign = _require_float64_array(
            core_state["skew_sign"], field="core_state.skew_sign", rank=1
        )
        training_scores = _require_float64_array(
            calibration["training_scores"],
            field="score_calibration.training_scores",
            rank=1,
        )
        if not x_sorted.shape[0] or not x_sorted.shape[1]:
            raise ECODCompositeAdapterError("ECOD empirical state must not be empty.")
        if skew_sign.shape != (x_sorted.shape[1],):
            raise ECODCompositeAdapterError(
                "ECOD skew_sign length must equal the feature dimension."
            )
        if training_scores.shape != (x_sorted.shape[0],):
            raise ECODCompositeAdapterError(
                "ECOD training_scores length must equal the training sample count."
            )
        if np.any(np.diff(x_sorted, axis=0) < 0.0):
            raise ECODCompositeAdapterError(
                "ECOD x_sorted columns must be monotonically non-decreasing."
            )
        if not np.isin(skew_sign, (-1.0, 0.0, 1.0)).all():
            raise ECODCompositeAdapterError("ECOD skew_sign values must be exactly -1, 0, or 1.")
        eps = core_state["eps"]
        if not isinstance(eps, (int, float)) or isinstance(eps, bool):
            raise ECODCompositeAdapterError("ECOD eps must be a finite number.")
        if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 1.0:
            raise ECODCompositeAdapterError("ECOD eps must be in (0, 1).")
        if calibration["kind"] != "empirical_training_scores_v1":
            raise ECODCompositeAdapterError("Unsupported ECOD score calibration kind.")
        if metadata["embedding_reduction"] != "auto_2d_v1":
            raise ECODCompositeAdapterError("Unsupported ECOD embedding reduction metadata.")
        if (
            not isinstance(metadata["feature_dimension"], int)
            or isinstance(metadata["feature_dimension"], bool)
            or int(metadata["feature_dimension"]) != int(x_sorted.shape[1])
        ):
            raise ECODCompositeAdapterError("ECOD feature_dimension metadata is invalid.")
        if (
            not isinstance(metadata["training_sample_count"], int)
            or isinstance(metadata["training_sample_count"], bool)
            or int(metadata["training_sample_count"]) != int(x_sorted.shape[0])
        ):
            raise ECODCompositeAdapterError("ECOD training_sample_count metadata is invalid.")
        total_bytes = int(x_sorted.nbytes + skew_sign.nbytes + training_scores.nbytes)
        if total_bytes > _MAX_STATE_BYTES:
            raise ECODCompositeAdapterError("ECOD fitted state exceeds its total byte limit.")

    def decode(self, detector: Any, state: Mapping[str, Any]) -> None:
        self.validate_state(state)
        core, chain = _core_chain(detector)
        core_state = state["core_state"]
        calibration = state["score_calibration"]
        x_sorted = np.array(core_state["x_sorted"], dtype=np.float64, copy=True)
        skew_sign = np.array(core_state["skew_sign"], dtype=np.float64, copy=True)
        training_scores = np.array(calibration["training_scores"], dtype=np.float64, copy=True)
        core._x_sorted = x_sorted
        core._skew_sign = skew_sign
        core.eps = float(core_state["eps"])
        core.decision_scores_ = np.array(training_scores, copy=True)
        for target in chain:
            if target is core:
                continue
            try:
                target.decision_scores_ = np.array(training_scores, copy=True)
            except Exception:
                pass
        if chain and chain[0] is not core and hasattr(chain[0], "_feature_extractor_fitted"):
            chain[0]._feature_extractor_fitted = True


@dataclass(frozen=True)
class EmbeddingComponentSpec:
    model_name: str
    format: ArtifactFormat
    source_path: Path
    source_size_bytes: int
    source_sha256: str
    external_data: tuple[Mapping[str, Any], ...]
    input_contract: Mapping[str, Any]
    output_contract: Mapping[str, Any]
    batch_size: int
    feature_dimension: int
    constructor_kwargs: Mapping[str, Any]
    allowed_providers: tuple[Mapping[str, Any], ...]
    verified_providers: tuple[Mapping[str, Any], ...]
    session_options: Mapping[str, Any]


def _finite_triplet(value: Any, *, field: str, nonzero: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ECODCompositeAdapterError(f"{field} must contain exactly three values.")
    normalized = [float(item) for item in value]
    if not all(math.isfinite(item) for item in normalized):
        raise ECODCompositeAdapterError(f"{field} values must be finite.")
    if nonzero and any(item == 0.0 for item in normalized):
        raise ECODCompositeAdapterError(f"{field} values must be non-zero.")
    return normalized


def _source_graph_path(extractor: Any) -> Path:
    checkpoint = getattr(extractor, "checkpoint", None)
    checkpoint_path = getattr(extractor, "checkpoint_path", None)
    if checkpoint is None or checkpoint_path is None or str(checkpoint) != str(checkpoint_path):
        raise ECODCompositeAdapterError(
            "Embedding extractor checkpoint/checkpoint_path must identify one exact graph."
        )
    path = Path(str(checkpoint_path)).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ECODCompositeAdapterError(
            f"Embedding graph must be a non-symlink regular file: {path}"
        )
    return path.resolve()


def _safe_dependency_location(value: str) -> str:
    import unicodedata
    from pathlib import PurePosixPath, PureWindowsPath

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
        raise ECODCompositeAdapterError(f"Unsafe ONNX external-data location: {location!r}.")
    posix = PurePosixPath(location)
    windows = PureWindowsPath(location)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ECODCompositeAdapterError(f"Unsafe ONNX external-data location: {location!r}.")
    if any(part in {"", ".", ".."} for part in location.split("/")):
        raise ECODCompositeAdapterError(f"Unsafe ONNX external-data location: {location!r}.")
    return location


def _snapshot_source_closure(
    source_path: Path,
    *,
    format: ArtifactFormat,
) -> tuple[int, str, tuple[Mapping[str, Any], ...]]:
    """Hash one stable no-follow snapshot of the graph dependency closure."""

    from pyimgano.artifacts.security import SecureSourceTree

    with tempfile.TemporaryDirectory(prefix="pyimgano-ecod-source-") as temporary:
        snapshot_root = Path(temporary)
        snapshot_graph = snapshot_root / source_path.name
        with SecureSourceTree(source_path.parent) as source_tree:
            size_bytes, sha256 = source_tree.copy_file(
                source_path.name,
                snapshot_graph,
            )
            if format is not ArtifactFormat.ONNX:
                return int(size_bytes), str(sha256), ()

            try:
                import onnx

                from pyimgano.artifacts.onnx_external_data import external_data_locations

                graph = onnx.load_model(str(snapshot_graph), load_external_data=False)
                locations = tuple(
                    _safe_dependency_location(item) for item in external_data_locations(graph)
                )
            except ECODCompositeAdapterError:
                raise
            except Exception as exc:  # noqa: BLE001 - copied protobuf trust boundary
                raise ECODCompositeAdapterError(
                    f"Cannot inspect ONNX embedding dependency closure: {exc}"
                ) from exc
            if any(item.casefold() == source_path.name.casefold() for item in locations):
                raise ECODCompositeAdapterError(
                    "ONNX external data must not overwrite the embedding graph."
                )
            dependencies: list[Mapping[str, Any]] = []
            for location in locations:
                target = snapshot_root.joinpath(*location.split("/"))
                dependency_size, dependency_sha = source_tree.copy_file(location, target)
                dependencies.append(
                    {
                        "location": location,
                        "size_bytes": int(dependency_size),
                        "sha256": str(dependency_sha),
                    }
                )
        return int(size_bytes), str(sha256), tuple(dependencies)


def _base_extractor(detector: Any, *, model_name: str) -> Any:
    extractor = getattr(detector, "_base_feature_extractor", None)
    if extractor is None:
        extractor = getattr(detector, "feature_extractor", None)
    if model_name == _ONNX_MODEL:
        from pyimgano.features.onnx_embed import ONNXEmbedExtractor

        expected = ONNXEmbedExtractor
    elif model_name == _TORCHSCRIPT_MODEL:
        from pyimgano.features.torchscript_embed import TorchscriptEmbedExtractor

        expected = TorchscriptEmbedExtractor
    else:  # pragma: no cover - guarded by caller
        raise ECODCompositeAdapterError(f"Unsupported model {model_name!r}.")
    if type(extractor) is not expected:
        raise ECODCompositeAdapterError(
            f"{model_name} export requires the canonical {expected.__name__}; "
            "custom/subclassed extractors are not representable by adapter schema v1."
        )
    if str(getattr(extractor, "input_color", "")).strip().lower() != "rgb":
        raise ECODCompositeAdapterError(
            "Composite artifact input is canonical RGB; input_color other than 'rgb' "
            "is not representable without changing the public input boundary."
        )
    return extractor


def _extractor_common(extractor: Any, *, input_name: str) -> tuple[dict[str, Any], int]:
    image_size = int(getattr(extractor, "image_size", 0))
    batch_size = int(getattr(extractor, "batch_size", 0))
    if image_size <= 0 or batch_size <= 0:
        raise ECODCompositeAdapterError("Embedding image_size and batch_size must be positive.")
    mean = _finite_triplet(getattr(extractor, "mean", None), field="embedding mean")
    std = _finite_triplet(getattr(extractor, "std", None), field="embedding std", nonzero=True)
    return (
        {
            "kind": "image_batch",
            "name": str(input_name),
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [image_size, image_size],
            "dynamic_axes": {"batch": True},
            "resize": {"mode": "stretch", "interpolation": "bilinear"},
            "scale": {"divisor": 255.0},
            "normalize": {"mean": mean, "std": std},
        },
        batch_size,
    )


def _feature_dimension(detector: Any) -> int:
    core, _chain = _core_chain(detector)
    x_sorted = getattr(core, "_x_sorted", None)
    if not isinstance(x_sorted, np.ndarray) or x_sorted.ndim != 2 or not x_sorted.shape[1]:
        raise ECODCompositeAdapterError("Fitted ECOD feature dimension is unavailable.")
    return int(x_sorted.shape[1])


def _embedding_component_spec(detector: Any, *, model_name: str) -> EmbeddingComponentSpec:
    expected_format = _expected_format(model_name)
    extractor = _base_extractor(detector, model_name=model_name)
    source_path = _source_graph_path(extractor)
    feature_dimension = _feature_dimension(detector)
    source_size_bytes, source_sha256, external_data = _snapshot_source_closure(
        source_path,
        format=expected_format,
    )

    if expected_format is ArtifactFormat.ONNX:
        ensure_ready = getattr(extractor, "_ensure_ready", None)
        if not callable(ensure_ready):  # pragma: no cover - exact type invariant
            raise ECODCompositeAdapterError("Canonical ONNX extractor cannot initialize.")
        ensure_ready()
        session = getattr(extractor, "_sess", None)
        input_name = str(getattr(extractor, "_input_name", "") or "").strip()
        output_name = str(getattr(extractor, "_output_name", "") or "").strip()
        if session is None or not input_name or not output_name:
            raise ECODCompositeAdapterError("ONNX extractor did not resolve its graph I/O.")
        if len(list(session.get_inputs())) != 1:
            raise ECODCompositeAdapterError(
                "Composite ONNX schema v1 requires exactly one runtime graph input."
            )
        input_contract, batch_size = _extractor_common(extractor, input_name=input_name)
        output_contract = {
            "kind": "feature_matrix",
            "name": output_name,
            "output_index": int(getattr(extractor, "output_index", 0)),
            "reduction": "auto_2d_v1",
        }
        providers = ({"name": "CPUExecutionProvider", "options": {}},)
        session_options: Mapping[str, Any] = {}
        embedding_name = "onnx_embed"
    else:
        input_contract, batch_size = _extractor_common(extractor, input_name="input")
        output_key = getattr(extractor, "output_key", None)
        output_contract = {
            "kind": "feature_matrix",
            "name": "embedding",
            "output_index": int(getattr(extractor, "output_index", 0)),
            "reduction": "auto_2d_v1",
        }
        if output_key is not None:
            if not isinstance(output_key, str) or not output_key:
                raise ECODCompositeAdapterError("TorchScript output_key must be non-empty.")
            output_contract["output_key"] = output_key
        providers = ({"name": "CPU", "options": {}},)
        session_options = {}
        embedding_name = "torchscript_embed"

    if int(output_contract["output_index"]) < 0:
        raise ECODCompositeAdapterError("Embedding output_index must not be negative.")
    core, _chain = _core_chain(detector)
    embedding_kwargs = {
        "device": "cpu",
        "batch_size": int(batch_size),
        "image_size": int(input_contract["size"][0]),
        "input_color": "rgb",
        "mean": list(input_contract["normalize"]["mean"]),
        "std": list(input_contract["normalize"]["std"]),
        "output_index": int(output_contract["output_index"]),
    }
    if expected_format is ArtifactFormat.ONNX:
        embedding_kwargs["input_name"] = str(input_contract["name"])
        embedding_kwargs["output_name"] = str(output_contract["name"])
        embedding_kwargs["providers"] = ["CPUExecutionProvider"]
    elif output_contract.get("output_key") is not None:
        embedding_kwargs["output_key"] = str(output_contract["output_key"])

    return EmbeddingComponentSpec(
        model_name=model_name,
        format=expected_format,
        source_path=source_path,
        source_size_bytes=source_size_bytes,
        source_sha256=source_sha256,
        external_data=external_data,
        input_contract=input_contract,
        output_contract=output_contract,
        batch_size=int(batch_size),
        feature_dimension=feature_dimension,
        constructor_kwargs={
            "contamination": float(getattr(detector, "contamination", 0.1)),
            "embedding_extractor": embedding_name,
            "embedding_kwargs": embedding_kwargs,
            "core_kwargs": {
                "n_jobs": int(getattr(core, "n_jobs", 1)),
                "eps": float(getattr(core, "eps", 1e-12)),
            },
        },
        allowed_providers=providers,
        verified_providers=providers,
        session_options=session_options,
    )


def _fixed_probe_images(image_size: int) -> tuple[np.ndarray, ...]:
    height = width = int(image_size)
    yy = np.arange(height, dtype=np.uint16)[:, None]
    xx = np.arange(width, dtype=np.uint16)[None, :]
    gradient = np.empty((height, width, 3), dtype=np.uint8)
    gradient[..., 0] = ((xx * 255) // max(width - 1, 1)).astype(np.uint8)
    gradient[..., 1] = ((yy * 255) // max(height - 1, 1)).astype(np.uint8)
    gradient[..., 2] = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
    checker = (((xx // 2 + yy // 2) % 2) * 255).astype(np.uint8)
    return (
        np.zeros((height, width, 3), dtype=np.uint8),
        gradient,
        np.stack([checker, np.roll(checker, 1, axis=0), np.roll(checker, 1, axis=1)], axis=-1),
    )


class ECODCompositeExportAdapter:
    """Certified exact-embedding-graph plus fitted-CoreECOD adapter."""

    adapter_id = ECOD_COMPOSITE_ADAPTER_ID
    adapter_version = ECOD_COMPOSITE_ADAPTER_VERSION
    model_names = _MODEL_NAMES
    state_codec_id = ECOD_CORE_CODEC_ID
    state_codec_version = ECOD_CORE_CODEC_VERSION
    state_schema_version = ECOD_CORE_STATE_SCHEMA_VERSION

    def inspect_source(self, source: Any) -> Mapping[str, Any]:
        model_name = ""
        class_name = type(source).__name__
        if class_name == "VisionONNXECOD":
            model_name = _ONNX_MODEL
        elif class_name == "VisionTorchscriptECOD":
            model_name = _TORCHSCRIPT_MODEL
        payload: dict[str, Any] = {"model_name": model_name, "fitted": False}
        try:
            core, _chain = _core_chain(source)
            payload["fitted"] = bool(
                getattr(core, "_x_sorted", None) is not None
                and getattr(core, "_skew_sign", None) is not None
            )
        except ECODCompositeAdapterError:
            pass
        return payload

    def declared_capability(self, format: ArtifactFormat) -> ExportCapability:
        if format in {ArtifactFormat.ONNX, ArtifactFormat.TORCHSCRIPT}:
            extra = "onnx_runtime" if format is ArtifactFormat.ONNX else "torch_runtime"
            return ExportCapability(
                format=format,
                status=ExportStatus.CONDITIONAL,
                layout=ExportLayout.COMPOSITE,
                conditions=(
                    "matching_canonical_embedding_extractor",
                    "complete_verified_checkpoint",
                    extra,
                ),
                reason_code="requires_concrete_composite_source",
                remediation=(
                    "Provide the fitted canonical embedding wrapper, its exact local graph, "
                    "and a checkpoint certified by this adapter."
                ),
            )
        return ExportCapability.unsupported(
            format,
            reason_code="format_not_certified",
            remediation="ECOD embedding composites support only their source ONNX/TorchScript format.",
        )

    def declared_capability_for_model(
        self,
        model_name: str,
        format: ArtifactFormat,
    ) -> ExportCapability:
        expected = _expected_format(str(model_name))
        if format is expected:
            return self.declared_capability(format)
        return ExportCapability.unsupported(
            format,
            reason_code="extractor_format_mismatch",
            remediation=(
                f"{model_name} can export only its exact {expected.value} embedding graph."
            ),
        )

    def effective_capability(
        self,
        format: ArtifactFormat,
        *,
        context: Mapping[str, Any] | NativeExportContext,
    ) -> ExportCapability:
        model_name = _model_name_from_context(context)
        if model_name not in _MODEL_NAMES:
            return ExportCapability.unsupported(format, reason_code="adapter_model_mismatch")
        declared = self.declared_capability_for_model(model_name, format)
        if _expected_format(model_name) is not format:
            return declared
        required = ("onnx", "onnxruntime") if format is ArtifactFormat.ONNX else ("torch",)
        missing = tuple(name for name in required if not _module_available(name))
        if missing:
            return ExportCapability.unsupported(
                format,
                reason_code="missing_export_dependency",
                remediation=f"Install the runtime extra; missing modules: {', '.join(missing)}.",
                availability=CapabilityAvailability.MISSING_EXTRA,
                conditions=declared.conditions,
            )
        if str(_context_mapping(context).get("phase", "")) == "pre_training":
            return declared
        contract = _checkpoint_from_context(context)
        if contract is None or not contract.strict_exportable:
            return ExportCapability.unsupported(
                format,
                reason_code="checkpoint_incomplete",
                remediation="Train through the registered checkpoint certification hook first.",
                conditions=declared.conditions,
            )
        mismatch = self._checkpoint_mismatch(contract)
        if mismatch is not None:
            return ExportCapability.unsupported(
                format,
                reason_code=mismatch,
                remediation="Recreate the checkpoint with the ECOD composite adapter.",
                conditions=declared.conditions,
            )
        return ExportCapability(
            format=format,
            status=ExportStatus.SUPPORTED,
            layout=ExportLayout.COMPOSITE,
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
            raise ECODCompositeAdapterError(
                "ECOD composite export requires a complete verified checkpoint."
            )
        mismatch = self._checkpoint_mismatch(contract)
        if mismatch is not None:
            raise ECODCompositeAdapterError(
                f"ECOD composite checkpoint contract is incompatible: {mismatch}."
            )

    def build_component_export_spec(
        self,
        detector: Any,
        *,
        format: ArtifactFormat,
        context: NativeExportContext | Mapping[str, Any],
    ) -> EmbeddingComponentSpec:
        model_name = _model_name_from_context(context)
        if _expected_format(model_name) is not format:
            raise ECODCompositeAdapterError(
                f"Requested {format.value} does not match {model_name}'s extractor."
            )
        return _embedding_component_spec(detector, model_name=model_name)

    def build_checkpoint_fingerprint_payload(
        self,
        detector: Any,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        values = dict(context or {})
        model_name = str(values.get("model_name", "")).strip()
        if model_name not in _MODEL_NAMES:
            class_name = type(detector).__name__
            model_name = _ONNX_MODEL if class_name == "VisionONNXECOD" else _TORCHSCRIPT_MODEL
        spec = _embedding_component_spec(detector, model_name=model_name)
        return {
            "schema": "pyimgano.ecod-composite-checkpoint-config.v1",
            "model_name": model_name,
            "format": spec.format.value,
            "embedding_graph": {
                "size_bytes": int(spec.source_size_bytes),
                "sha256": str(spec.source_sha256),
                "external_data": [dict(item) for item in spec.external_data],
            },
            "input_contract": dict(spec.input_contract),
            "embedding_output_contract": dict(spec.output_contract),
            "batch_size": int(spec.batch_size),
            "feature_dimension": int(spec.feature_dimension),
            "constructor_kwargs": dict(spec.constructor_kwargs),
        }

    def validate_checkpoint_source_binding(
        self,
        detector: Any,
        contract: CheckpointContract,
        *,
        context: NativeExportContext | Mapping[str, Any],
    ) -> None:
        from pyimgano.artifacts import canonical_json_bytes

        payload = self.build_checkpoint_fingerprint_payload(
            detector,
            context={"model_name": _model_name_from_context(context)},
        )
        expected = "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if contract.model_config_fingerprint != expected:
            raise ECODCompositeAdapterError(
                "Certified checkpoint is not bound to this exact embedding graph/configuration."
            )

    def build_fresh_restore_detector(
        self,
        original: Any,
        *,
        context: Mapping[str, Any],
    ) -> Any:
        model_name = str(context.get("model_name", "")).strip()
        spec = _embedding_component_spec(original, model_name=model_name)
        constructor = dict(spec.constructor_kwargs)
        embedding_kwargs = dict(constructor["embedding_kwargs"])
        embedding_kwargs["checkpoint_path"] = str(spec.source_path)
        constructor["embedding_kwargs"] = embedding_kwargs
        import pyimgano.models  # noqa: F401 - populate registry
        from pyimgano.models.registry import create_model

        return create_model(model_name, **constructor)

    def build_probe_spec(
        self,
        detector: Any,
        *,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> ProbeSpec:
        model_name = (
            _model_name_from_context(context)
            if context is not None
            else (
                _ONNX_MODEL if type(detector).__name__ == "VisionONNXECOD" else _TORCHSCRIPT_MODEL
            )
        )
        spec = _embedding_component_spec(detector, model_name=model_name)
        return ProbeSpec(
            inputs=_fixed_probe_images(int(spec.input_contract["size"][0])),
            expected_outputs=("score",),
            absolute_tolerance=1e-6,
            relative_tolerance=1e-5,
        )

    def evaluate_probe(
        self,
        detector: Any,
        spec: ProbeSpec | None = None,
    ) -> Mapping[str, np.ndarray]:
        selected = spec if spec is not None else self.build_probe_spec(detector)
        decision = getattr(detector, "decision_function", None)
        if not callable(decision):
            raise ECODCompositeAdapterError("Detector exposes no decision_function().")
        scores = np.asarray(decision(list(selected.inputs)), dtype=np.float64).reshape(-1)
        if scores.shape != (len(selected.inputs),) or not np.isfinite(scores).all():
            raise ECODCompositeAdapterError("ECOD probe returned invalid scores.")
        return {"score": scores}

    def verify_roundtrip(
        self,
        original: Any,
        restored: Any,
        spec: ProbeSpec | None = None,
    ) -> Mapping[str, Any]:
        selected = spec if spec is not None else self.build_probe_spec(original)
        before = np.asarray(self.evaluate_probe(original, selected)["score"], dtype=np.float64)
        after = np.asarray(self.evaluate_probe(restored, selected)["score"], dtype=np.float64)
        maximum = float(np.max(np.abs(after - before), initial=0.0))
        passed = bool(
            np.allclose(
                before,
                after,
                atol=selected.absolute_tolerance,
                rtol=selected.relative_tolerance,
            )
        )
        report = {
            "passed": passed,
            "max_score_absolute_error": maximum,
            "sample_count": len(selected.inputs),
        }
        if not passed:
            raise ECODCompositeAdapterError(
                f"ECOD composite checkpoint round-trip parity failed: {report!r}."
            )
        return report

    def load_composite_core(
        self,
        state_path: str | Path,
        *,
        model_name: str,
        codec_id: str,
        codec_version: int,
    ) -> Any:
        if codec_id != self.state_codec_id or int(codec_version) != self.state_codec_version:
            raise ECODCompositeAdapterError("Composite core codec identity is incompatible.")
        from pyimgano.exporting.state_codec import load_fitted_state
        from pyimgano.models.ecod import CoreECOD

        core = CoreECOD()
        load_fitted_state(core, state_path, expected_model_name=model_name)
        return core

    def compose(
        self,
        *,
        component_runtime: Any,
        fitted_core: Any,
        inputs: Sequence[Any],
        include_maps: bool,
    ) -> tuple[np.ndarray, None]:
        del include_maps
        extract = getattr(component_runtime, "extract", None)
        if not callable(extract):
            raise ECODCompositeAdapterError("Embedding component exposes no extract().")
        features = np.asarray(extract(list(inputs)), dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != len(inputs):
            raise ECODCompositeAdapterError(
                "Embedding component must return one feature row per image."
            )
        expected = getattr(fitted_core, "_x_sorted", None)
        expected_dimension = (
            int(expected.shape[1]) if isinstance(expected, np.ndarray) and expected.ndim == 2 else 0
        )
        if not expected_dimension or int(features.shape[1]) != expected_dimension:
            raise ECODCompositeAdapterError(
                "Embedding feature dimension does not match the fitted ECOD state."
            )
        return np.asarray(fitted_core.decision_function(features), dtype=np.float64), None

    def build_output_contract(self) -> Mapping[str, Any]:
        return {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        }

    def build_runtime_spec(
        self,
        *,
        format: ArtifactFormat,
        context: NativeExportContext | Mapping[str, Any],
    ) -> Mapping[str, Any]:
        model_name = _model_name_from_context(context)
        expected = _expected_format(model_name)
        if format is not expected:
            raise ECODCompositeAdapterError("Runtime format does not match the source extractor.")
        provider = "CPUExecutionProvider" if format is ArtifactFormat.ONNX else "CPU"
        return {
            "backend": "pyimgano",
            "allowed_providers": [{"name": provider, "options": {}}],
            "verified_providers": [{"name": provider, "options": {}}],
            "composition_adapter": {
                "id": self.adapter_id,
                "version": self.adapter_version,
            },
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
        from pyimgano.exporting.exporters.composite import export_composite

        return export_composite(
            detector,
            format=format,
            context=context,
            out=out,
            adapter=self,
            overwrite=overwrite,
        )


ECOD_CORE_STATE_CODEC = CoreECODStateCodec()
ECOD_COMPOSITE_EXPORT_ADAPTER = ECODCompositeExportAdapter()


__all__ = [
    "ECOD_COMPOSITE_ADAPTER_ID",
    "ECOD_COMPOSITE_ADAPTER_VERSION",
    "ECOD_COMPOSITE_EXPORT_ADAPTER",
    "ECOD_CORE_CODEC_ID",
    "ECOD_CORE_CODEC_VERSION",
    "ECOD_CORE_STATE_CODEC",
    "ECOD_CORE_STATE_SCHEMA_VERSION",
    "CoreECODStateCodec",
    "ECODCompositeAdapterError",
    "ECODCompositeExportAdapter",
    "EmbeddingComponentSpec",
]
