from __future__ import annotations

"""Explicit migration loader for pre-schema-v1 run/config artifacts."""

import warnings
from pathlib import Path
from typing import Any, Literal


class LegacyArtifactWarning(UserWarning):
    """Warn that a legacy source lacks schema-v1 closure and identity guarantees."""


def load_legacy_artifact(
    source: str | Path,
    *,
    kind: Literal["run", "infer_config"],
    allow_legacy: bool = False,
    category: str | None = None,
    device: str | None = None,
    trust_checkpoint: bool = False,
) -> Any:
    """Load a legacy run or infer config through its original explicit contract.

    This helper never auto-detects a source kind and never accepts a raw ONNX file.
    It exists only as an opt-in migration bridge; new deployments should use
    :func:`load_artifact`, whose schema-v1 loader verifies the complete file closure.
    """

    if not allow_legacy:
        raise ValueError(
            "Legacy artifact loading requires allow_legacy=True. Prefer exporting "
            "a schema-v1 artifact with `pyimgano export --from-run ...`."
        )
    normalized_kind = str(kind).strip().lower().replace("-", "_")
    if normalized_kind not in {"run", "infer_config"}:
        raise ValueError(
            "kind must be exactly 'run' or 'infer_config'; path sniffing is forbidden."
        )

    warnings.warn(
        "Loading a legacy source without schema-v1 dependency closure. Migrate it with "
        "`pyimgano export` when a certified adapter is available.",
        LegacyArtifactWarning,
        stacklevel=2,
    )

    import pyimgano.services.infer_context_service as infer_context_service
    import pyimgano.services.infer_load_service as infer_load_service

    if normalized_kind == "run":
        context = infer_context_service.prepare_from_run_context(
            infer_context_service.FromRunInferContextRequest(
                run_dir=str(source),
                from_run_category=(str(category) if category is not None else None),
                device=(str(device) if device is not None else None),
            )
        )
    else:
        context = infer_context_service.prepare_infer_config_context(
            infer_context_service.InferConfigContextRequest(
                config_path=str(source),
                infer_category=(str(category) if category is not None else None),
                device=(str(device) if device is not None else None),
            )
        )
    loaded = infer_load_service.load_config_backed_infer_detector(
        infer_load_service.ConfigBackedInferLoadRequest(
            context=context,
            trust_checkpoint=bool(trust_checkpoint),
        )
    )
    return loaded.detector


__all__ = ["LegacyArtifactWarning", "load_legacy_artifact"]
