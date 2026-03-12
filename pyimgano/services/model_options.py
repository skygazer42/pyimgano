from __future__ import annotations

from typing import Any, Callable, Mapping

from pyimgano.models.model_kwargs import (
    build_model_kwargs,
    get_model_signature_info,
    merge_checkpoint_path,
)
from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.presets.catalog import resolve_model_preset

from pyimgano.utils.optional_deps import optional_import


def _faiss_available() -> bool:
    module, _error = optional_import("faiss")
    return module is not None


def _default_knn_backend() -> str:
    return "faiss" if _faiss_available() else "sklearn"


def _looks_like_onnx_route(model_name: str, user_kwargs: Mapping[str, Any]) -> bool:
    name = str(model_name)
    if "onnx" in name:
        return True
    if str(user_kwargs.get("embedding_extractor", "")).strip() == "onnx_embed":
        return True
    fx = user_kwargs.get("feature_extractor", None)
    if fx == "onnx_embed":
        return True
    if isinstance(fx, Mapping) and str(fx.get("name", "")).strip() == "onnx_embed":
        return True
    return False


def apply_onnx_session_options_shorthand(
    *,
    model_name: str,
    user_kwargs: dict[str, Any],
    session_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Best-effort injection of ORT session options into model kwargs.

    Supported targets:
    - first-class `session_options` model kwarg on ONNX wrappers
    - `embedding_kwargs.session_options` for embedding-core style models
    - `feature_extractor.kwargs.session_options` for feature pipeline specs
    """

    if not session_options:
        return dict(user_kwargs)

    if not _looks_like_onnx_route(model_name, user_kwargs):
        raise ValueError(
            "--onnx-session-options is only supported for ONNX-based routes "
            "(e.g. vision_onnx_* or embedding_extractor='onnx_embed')."
        )

    accepted, accepts_var_kwargs = get_model_signature_info(str(model_name))
    out = dict(user_kwargs)

    if accepts_var_kwargs or ("session_options" in accepted):
        base = out.get("session_options", None)
        merged = dict(base) if isinstance(base, Mapping) else {}
        merged.update(dict(session_options))
        out["session_options"] = merged
        return out

    if "embedding_kwargs" in accepted:
        embedder = out.get("embedding_extractor", None)
        if embedder is not None and str(embedder) != "onnx_embed":
            raise ValueError(
                "--onnx-session-options requires embedding_extractor='onnx_embed' when targeting embedding_kwargs."
            )
        ek = out.get("embedding_kwargs", None)
        ek_dict = dict(ek) if isinstance(ek, Mapping) else {}
        base = ek_dict.get("session_options", None)
        merged = dict(base) if isinstance(base, Mapping) else {}
        merged.update(dict(session_options))
        ek_dict["session_options"] = merged
        out["embedding_kwargs"] = ek_dict
        return out

    if "feature_extractor" in accepted:
        fx = out.get("feature_extractor", None)
        if fx == "onnx_embed":
            out["feature_extractor"] = {
                "name": "onnx_embed",
                "kwargs": {"session_options": dict(session_options)},
            }
            return out
        if isinstance(fx, Mapping) and str(fx.get("name", "")).strip() == "onnx_embed":
            fx_kwargs = fx.get("kwargs", None)
            fx_kwargs_dict = dict(fx_kwargs) if isinstance(fx_kwargs, Mapping) else {}
            base = fx_kwargs_dict.get("session_options", None)
            merged = dict(base) if isinstance(base, Mapping) else {}
            merged.update(dict(session_options))
            fx_kwargs_dict["session_options"] = merged
            out["feature_extractor"] = {"name": "onnx_embed", "kwargs": fx_kwargs_dict}
            return out

    raise ValueError(
        "--onnx-session-options could not be applied to this model. "
        "Use --model-kwargs to pass session_options directly."
    )


def resolve_requested_model(requested_model: str) -> tuple[str, dict[str, Any], Any]:
    import pyimgano.models  # noqa: F401

    model_name = str(requested_model)
    preset_model_auto_kwargs: dict[str, Any] = {}

    try:
        entry = MODEL_REGISTRY.info(model_name)
    except KeyError as exc:
        preset = resolve_model_preset(model_name)
        if preset is None:
            raise ValueError(
                f"Unknown model or model preset: {requested_model!r}. "
                "Use --list-models/--list-model-presets for discovery."
            ) from exc

        model_name = str(preset.model)
        preset_model_auto_kwargs = dict(preset.kwargs)
        try:
            entry = MODEL_REGISTRY.info(model_name)
        except KeyError as exc2:
            raise ValueError(
                f"Model preset {requested_model!r} refers to unknown model: {model_name!r}"
            ) from exc2

    return model_name, preset_model_auto_kwargs, entry


def enforce_checkpoint_requirement(
    *,
    model_name: str,
    model_kwargs: Mapping[str, Any],
    trained_checkpoint_path: str | None = None,
    extra_guidance: str | None = None,
) -> None:
    import pyimgano.models  # noqa: F401

    entry = MODEL_REGISTRY.info(str(model_name))
    requires = bool(entry.metadata.get("requires_checkpoint", False))
    if not requires:
        return

    has_kwarg = model_kwargs.get("checkpoint_path", None) is not None
    has_trained = trained_checkpoint_path is not None
    if has_kwarg or has_trained:
        return

    message = (
        f"Model {model_name!r} requires a checkpoint. "
        "Provide --checkpoint-path (or set checkpoint_path in --model-kwargs)."
    )
    if extra_guidance:
        message = f"{message} {str(extra_guidance).strip()}"
    raise ValueError(message)


def resolve_preset_kwargs(
    preset: str | None,
    model_name: str,
    *,
    default_knn_backend: Callable[[], str] | None = None,
) -> dict[str, Any]:
    if preset is None:
        return {}

    knn_backend = default_knn_backend or _default_knn_backend

    if preset == "industrial-fast":
        if model_name == "vision_patchcore":
            return {
                "backbone": "resnet50",
                "coreset_sampling_ratio": 0.02,
                "feature_projection_dim": 256,
                "n_neighbors": 3,
                "knn_backend": knn_backend(),
                "memory_bank_dtype": "float16",
            }
        if model_name == "vision_padim":
            return {
                "backbone": "resnet18",
                "d_reduced": 32,
                "image_size": 192,
            }
        if model_name == "vision_spade":
            return {
                "backbone": "resnet18",
                "image_size": 192,
                "k_neighbors": 20,
                "feature_levels": ["layer2", "layer3"],
                "gaussian_sigma": 2.0,
            }
        if model_name == "vision_anomalydino":
            return {
                "knn_backend": knn_backend(),
                "coreset_sampling_ratio": 0.1,
                "image_size": 336,
            }
        if model_name == "vision_softpatch":
            return {
                "knn_backend": knn_backend(),
                "coreset_sampling_ratio": 0.1,
                "train_patch_outlier_quantile": 0.1,
                "image_size": 336,
            }
        if model_name == "vision_simplenet":
            return {
                "backbone": "resnet50",
                "epochs": 5,
                "batch_size": 16,
            }
        if model_name == "vision_fastflow":
            return {
                "epoch_num": 5,
                "n_flow_steps": 4,
                "batch_size": 32,
            }
        if model_name == "vision_cflow":
            return {
                "epochs": 10,
                "n_flows": 2,
                "batch_size": 32,
            }
        if model_name == "vision_stfpm":
            return {
                "epochs": 10,
                "batch_size": 32,
            }
        if model_name in ("vision_reverse_distillation", "vision_reverse_dist"):
            return {
                "epoch_num": 5,
                "batch_size": 32,
            }
        if model_name == "vision_draem":
            return {
                "image_size": 256,
                "epochs": 20,
                "batch_size": 16,
            }
        return {}

    if preset == "industrial-balanced":
        if model_name == "vision_patchcore":
            return {
                "backbone": "resnet50",
                "coreset_sampling_ratio": 0.05,
                "feature_projection_dim": 512,
                "n_neighbors": 5,
                "knn_backend": knn_backend(),
                "memory_bank_dtype": "float16",
            }
        if model_name == "vision_padim":
            return {
                "backbone": "resnet18",
                "d_reduced": 64,
                "image_size": 224,
            }
        if model_name == "vision_spade":
            return {
                "backbone": "resnet50",
                "image_size": 256,
                "k_neighbors": 50,
                "feature_levels": ["layer1", "layer2", "layer3"],
                "gaussian_sigma": 4.0,
            }
        if model_name == "vision_anomalydino":
            return {
                "knn_backend": knn_backend(),
                "coreset_sampling_ratio": 0.2,
                "image_size": 448,
            }
        if model_name == "vision_softpatch":
            return {
                "knn_backend": knn_backend(),
                "coreset_sampling_ratio": 0.2,
                "train_patch_outlier_quantile": 0.1,
                "image_size": 448,
            }
        if model_name == "vision_simplenet":
            return {
                "backbone": "resnet50",
                "epochs": 10,
                "batch_size": 16,
            }
        if model_name == "vision_fastflow":
            return {
                "epoch_num": 10,
                "n_flow_steps": 6,
                "batch_size": 32,
            }
        if model_name == "vision_cflow":
            return {
                "epochs": 15,
                "n_flows": 4,
                "batch_size": 32,
            }
        if model_name == "vision_stfpm":
            return {
                "epochs": 50,
                "batch_size": 32,
            }
        if model_name in ("vision_reverse_distillation", "vision_reverse_dist"):
            return {
                "epoch_num": 10,
                "batch_size": 32,
            }
        if model_name == "vision_draem":
            return {
                "image_size": 256,
                "epochs": 50,
                "batch_size": 16,
            }
        return {}

    if preset == "industrial-accurate":
        if model_name == "vision_patchcore":
            return {
                "backbone": "wide_resnet50",
                "coreset_sampling_ratio": 0.1,
                "n_neighbors": 9,
                "knn_backend": knn_backend(),
            }
        if model_name == "vision_padim":
            return {
                "backbone": "resnet50",
                "d_reduced": 128,
                "image_size": 224,
            }
        if model_name == "vision_spade":
            return {
                "backbone": "wide_resnet50",
                "image_size": 256,
                "k_neighbors": 50,
                "feature_levels": ["layer1", "layer2", "layer3"],
                "gaussian_sigma": 4.0,
            }
        if model_name == "vision_anomalydino":
            return {
                "knn_backend": knn_backend(),
                "coreset_sampling_ratio": 0.5,
                "image_size": 518,
            }
        if model_name == "vision_softpatch":
            return {
                "knn_backend": knn_backend(),
                "coreset_sampling_ratio": 0.5,
                "train_patch_outlier_quantile": 0.05,
                "image_size": 518,
            }
        if model_name == "vision_simplenet":
            return {
                "backbone": "wide_resnet50",
                "epochs": 20,
                "batch_size": 16,
            }
        if model_name == "vision_fastflow":
            return {
                "epoch_num": 20,
                "n_flow_steps": 8,
                "batch_size": 32,
            }
        if model_name == "vision_cflow":
            return {
                "epochs": 50,
                "n_flows": 8,
                "batch_size": 16,
            }
        if model_name == "vision_stfpm":
            return {
                "epochs": 100,
                "batch_size": 32,
            }
        if model_name in ("vision_reverse_distillation", "vision_reverse_dist"):
            return {
                "epoch_num": 20,
                "batch_size": 32,
            }
        if model_name == "vision_draem":
            return {
                "image_size": 256,
                "epochs": 100,
                "batch_size": 16,
            }
        return {}

    raise ValueError(
        f"Unknown preset: {preset!r}. Choose from: "
        "industrial-fast, industrial-balanced, industrial-accurate"
    )


def resolve_model_options(
    *,
    model_name: str,
    preset: str | None,
    user_kwargs: dict[str, Any],
    auto_kwargs: dict[str, Any],
    checkpoint_path: str | None,
    default_knn_backend: Callable[[], str] | None = None,
) -> dict[str, Any]:
    merged_user = merge_checkpoint_path(user_kwargs, checkpoint_path=checkpoint_path)
    preset_kwargs = resolve_preset_kwargs(
        preset,
        model_name,
        default_knn_backend=default_knn_backend,
    )
    return build_model_kwargs(
        model_name,
        user_kwargs=merged_user,
        preset_kwargs=preset_kwargs,
        auto_kwargs=auto_kwargs,
    )


__all__ = [
    "apply_onnx_session_options_shorthand",
    "enforce_checkpoint_requirement",
    "resolve_model_options",
    "resolve_preset_kwargs",
    "resolve_requested_model",
]
