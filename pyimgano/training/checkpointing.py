from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pyimgano.exporting.types import (
    CheckpointCompleteness,
    CheckpointContract,
    SerializationKind,
)


def save_checkpoint(detector: Any, path: str | Path) -> Path:
    """Best-effort checkpoint saving for recipe/workbench runs.

    Priority:
    1) `detector.save_checkpoint(path)` when present.
    2) `detector.model.state_dict()` saved via `torch.save(...)` when present.
    3) store its structured state in a non-executable safe archive.
    4) serialize the detector object itself via joblib when necessary.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from pyimgano.inference.runtime_wrappers import unwrap_runtime_detector

    checkpoint_target = unwrap_runtime_detector(detector)

    save_fn = getattr(checkpoint_target, "save_checkpoint", None)
    if callable(save_fn):
        try:
            save_fn(str(out_path))
        except TypeError:
            save_fn(out_path)
        return out_path

    model = getattr(checkpoint_target, "model", None)
    if model is not None and callable(getattr(model, "state_dict", None)):
        from pyimgano.utils.optional_deps import require

        torch = require(
            "torch",
            extra="torch",
            purpose="save checkpoints via detector.model.state_dict",
        )

        state = model.state_dict()
        torch.save(state, out_path)
        return out_path

    try:
        from pyimgano.serialization.safe_checkpoint import SafeCheckpointError
        from pyimgano.serialization.safe_detector_state import save_safe_detector_state

        return save_safe_detector_state(checkpoint_target, out_path)
    except SafeCheckpointError:
        pass

    try:
        from pyimgano.models.serialization import save_model

        return save_model(checkpoint_target, out_path)
    except Exception as exc:
        raise NotImplementedError(
            "Detector does not support checkpoint saving. Expected one of:\n"
            "- `detector.save_checkpoint(path)`\n"
            "- `detector.model` with a torch-style `state_dict()`\n"
            "- a detector state containing safe structured values\n"
            "- a joblib-serializable detector object\n"
        ) from exc


def inspect_checkpoint_contract(path: str | Path) -> CheckpointContract:
    """Return persisted checkpoint evidence without upgrading unknown state."""

    from pyimgano.exporting.state_codec import inspect_checkpoint_contract as inspect

    return inspect(path)


def build_checkpoint_contract(
    path: str | Path,
    *,
    codec_id: str,
    codec_version: int,
    adapter_id: str,
    adapter_version: int,
    model_config: Mapping[str, Any],
    state_schema_version: int,
    roundtrip_verified: bool,
    roundtrip: Mapping[str, Any],
    serialization: SerializationKind = SerializationKind.SAFE_DATA,
    requires_trust: bool = False,
) -> CheckpointContract:
    """Build explicit evidence after save/probe; never infer it from loading."""

    import hashlib

    from pyimgano.artifacts import canonical_json_bytes
    from pyimgano.exporting.writer import sha256_file

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    fingerprint = "sha256:" + hashlib.sha256(canonical_json_bytes(dict(model_config))).hexdigest()
    return CheckpointContract(
        completeness=(
            CheckpointCompleteness.COMPLETE
            if bool(roundtrip_verified)
            else CheckpointCompleteness.PARTIAL
        ),
        codec_id=str(codec_id),
        codec_version=int(codec_version),
        adapter_id=str(adapter_id),
        adapter_version=int(adapter_version),
        model_config_fingerprint=fingerprint,
        state_schema_version=int(state_schema_version),
        serialization=serialization,
        requires_trust=bool(requires_trust),
        size_bytes=int(checkpoint.stat().st_size),
        sha256=sha256_file(checkpoint),
        roundtrip_verified=bool(roundtrip_verified),
        roundtrip=dict(roundtrip),
    )


def failed_checkpoint_contract(reason: str) -> CheckpointContract:
    return CheckpointContract(
        completeness=CheckpointCompleteness.FAILED,
        roundtrip_verified=False,
        failure_reason=str(reason),
    )


__all__ = [
    "build_checkpoint_contract",
    "failed_checkpoint_contract",
    "inspect_checkpoint_contract",
    "save_checkpoint",
]
