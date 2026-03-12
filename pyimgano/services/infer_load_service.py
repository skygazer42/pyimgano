from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pyimgano.models.registry import create_model
from pyimgano.services.infer_context_service import ConfigBackedInferContext
from pyimgano.services.model_options import (
    enforce_checkpoint_requirement,
    resolve_model_options,
    resolve_requested_model,
)
import pyimgano.services.workbench_run_service as workbench_run_service


@dataclass(frozen=True)
class DirectInferLoadRequest:
    requested_model: str
    preset: str | None = None
    device: str = "cpu"
    contamination: float = 0.1
    pretrained: bool = False
    seed: int | None = None
    user_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class DirectInferLoadResult:
    model_name: str
    detector: Any
    model_kwargs: dict[str, Any]


@dataclass(frozen=True)
class ConfigBackedInferLoadRequest:
    context: ConfigBackedInferContext
    seed: int | None = None
    user_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConfigBackedInferLoadResult:
    model_name: str
    detector: Any
    model_kwargs: dict[str, Any]


def load_direct_infer_detector(
    request: DirectInferLoadRequest,
    *,
    create_detector: Callable[..., Any] | None = None,
) -> DirectInferLoadResult:
    model_name, preset_model_auto_kwargs, _entry = resolve_requested_model(
        str(request.requested_model)
    )

    auto_kwargs: dict[str, Any] = dict(preset_model_auto_kwargs)
    auto_kwargs.update(
        {
            "device": str(request.device),
            "contamination": float(request.contamination),
            "pretrained": bool(request.pretrained),
        }
    )
    if request.seed is not None:
        auto_kwargs["random_seed"] = int(request.seed)
        auto_kwargs["random_state"] = int(request.seed)

    model_kwargs = resolve_model_options(
        model_name=model_name,
        preset=(str(request.preset) if request.preset is not None else None),
        user_kwargs=dict(request.user_kwargs or {}),
        auto_kwargs=auto_kwargs,
        checkpoint_path=None,
    )
    enforce_checkpoint_requirement(
        model_name=model_name,
        model_kwargs=dict(model_kwargs),
        trained_checkpoint_path=None,
        extra_guidance="Or load via --from-run/--infer-config that includes a checkpoint.",
    )

    detector_factory = create_detector or create_model
    detector = detector_factory(model_name, **model_kwargs)
    return DirectInferLoadResult(
        model_name=model_name,
        detector=detector,
        model_kwargs=dict(model_kwargs),
    )


def load_config_backed_infer_detector(
    request: ConfigBackedInferLoadRequest,
    *,
    create_detector: Callable[..., Any] | None = None,
    load_checkpoint: Callable[[Any, str], None] | None = None,
) -> ConfigBackedInferLoadResult:
    context = request.context
    model_name = str(context.model_name)

    auto_kwargs: dict[str, Any] = {
        "device": str(context.device),
        "contamination": float(context.contamination),
        "pretrained": bool(context.pretrained),
    }
    if request.seed is not None:
        auto_kwargs["random_seed"] = int(request.seed)
        auto_kwargs["random_state"] = int(request.seed)

    model_kwargs = resolve_model_options(
        model_name=model_name,
        preset=(str(context.preset) if context.preset is not None else None),
        user_kwargs=dict(request.user_kwargs or {}),
        auto_kwargs=auto_kwargs,
        checkpoint_path=None,
    )
    enforce_checkpoint_requirement(
        model_name=model_name,
        model_kwargs=dict(model_kwargs),
        trained_checkpoint_path=(
            str(context.trained_checkpoint_path)
            if context.trained_checkpoint_path is not None
            else None
        ),
        extra_guidance="Or load via --from-run/--infer-config that includes a checkpoint.",
    )

    detector_factory = create_detector or create_model
    detector = detector_factory(model_name, **model_kwargs)

    load_checkpoint_fn = load_checkpoint
    if load_checkpoint_fn is None:
        load_checkpoint_fn = workbench_run_service.load_checkpoint_into_detector

    if context.trained_checkpoint_path is not None:
        load_checkpoint_fn(detector, str(context.trained_checkpoint_path))
    if context.threshold is not None:
        setattr(detector, "threshold_", float(context.threshold))

    return ConfigBackedInferLoadResult(
        model_name=model_name,
        detector=detector,
        model_kwargs=dict(model_kwargs),
    )


__all__ = [
    "ConfigBackedInferLoadRequest",
    "ConfigBackedInferLoadResult",
    "DirectInferLoadRequest",
    "DirectInferLoadResult",
    "load_config_backed_infer_detector",
    "load_direct_infer_detector",
]
