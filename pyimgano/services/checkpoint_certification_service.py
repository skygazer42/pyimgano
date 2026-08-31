from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class CheckpointCertificationError(RuntimeError):
    """Raised when a registered export adapter cannot prove checkpoint parity."""


def _resolved_model_kwargs(config: Any) -> dict[str, Any]:
    from pyimgano.services.model_options import resolve_model_options

    user_kwargs = dict(getattr(config.model, "model_kwargs", {}) or {})
    for key in ("checkpoint", "checkpoint_path"):
        user_kwargs.pop(key, None)
    auto_kwargs: dict[str, Any] = {
        "device": str(config.model.device),
        "contamination": float(config.model.contamination),
        # A complete fitted-state codec must make pretrained initialization
        # irrelevant. Keeping this false prevents an export restore from downloading.
        "pretrained": False,
    }
    if getattr(config, "seed", None) is not None:
        auto_kwargs["random_seed"] = int(config.seed)
        auto_kwargs["random_state"] = int(config.seed)
    return dict(
        resolve_model_options(
            model_name=str(config.model.name),
            preset=(str(config.model.preset) if config.model.preset is not None else None),
            user_kwargs=user_kwargs,
            auto_kwargs=auto_kwargs,
            checkpoint_path=None,
        )
    )


def _fresh_detector(config: Any) -> Any:
    import pyimgano.models  # noqa: F401 - populate lazy registry
    from pyimgano.models.registry import create_model

    return create_model(str(config.model.name), **_resolved_model_kwargs(config))


def _fresh_restore_detector(
    adapter: Any,
    *,
    original: Any,
    config: Any,
    context: Mapping[str, Any],
) -> Any:
    build = getattr(adapter, "build_fresh_restore_detector", None)
    if callable(build):
        return build(original, context=context)
    return _fresh_detector(config)


def _checkpoint_fingerprint_payload(
    adapter: Any,
    *,
    original: Any,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    build = getattr(adapter, "build_checkpoint_fingerprint_payload", None)
    if callable(build):
        payload = build(original, context=context)
        if not isinstance(payload, Mapping):
            raise CheckpointCertificationError(
                "Export adapter checkpoint fingerprint payload must be a mapping."
            )
        return dict(payload)
    model_kwargs = context.get("model_kwargs")
    if not isinstance(model_kwargs, Mapping):
        raise CheckpointCertificationError(
            "Checkpoint certification context is missing model_kwargs."
        )
    return dict(model_kwargs)


def _normalize_roundtrip_result(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif isinstance(value, bool):
        payload = {"passed": bool(value)}
    elif value is None:
        payload = {"passed": True}
    else:
        raise CheckpointCertificationError(
            "Export adapter verify_roundtrip() must return a mapping, bool, or None."
        )
    if payload.get("passed") is not True:
        raise CheckpointCertificationError(
            f"Export adapter checkpoint parity probe failed: {payload!r}"
        )
    return payload


def _fallback_state_roundtrip(codec: Any, original: Any, restored: Any) -> dict[str, Any]:
    import numpy as np

    original_state = dict(codec.encode(original))
    restored_state = dict(codec.encode(restored))
    if set(original_state) != set(restored_state):
        raise CheckpointCertificationError("Restored fitted-state field names do not match.")
    for name in sorted(original_state):
        left = original_state[name]
        right = restored_state[name]
        try:
            np.testing.assert_equal(np.asarray(left), np.asarray(right))
        except Exception as exc:
            raise CheckpointCertificationError(
                f"Restored fitted-state field differs: {name!r}."
            ) from exc
    return {"passed": True, "kind": "exact_state_roundtrip", "fields": sorted(original_state)}


def _verify_roundtrip(
    adapter: Any,
    *,
    original: Any,
    restored: Any,
    codec: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    build_probe = getattr(adapter, "build_probe_spec", None)
    verify = getattr(adapter, "verify_roundtrip", None)
    if callable(build_probe) and callable(verify):
        spec = build_probe(original, context=context)
        result = _normalize_roundtrip_result(verify(original, restored, spec))
        result.setdefault("kind", "adapter_score_map_parity")
        return result
    return _fallback_state_roundtrip(codec, original, restored)


def _load_builtin_adapters() -> None:
    try:
        import pyimgano.exporting.adapters  # noqa: F401
    except ImportError:
        # No built-in adapter package is a valid state for installations that only
        # consume artifacts. Registry lookup below remains the authority.
        pass


def certify_checkpoint_for_export(
    detector: Any,
    checkpoint_path: str | Path,
    *,
    config: Any,
    probe_inputs: Sequence[Any] = (),
) -> Any | None:
    """Replace a just-saved checkpoint with certified safe fitted state.

    Returns a complete ``CheckpointContract`` for registered adapters, ``None``
    when the model has no adapter, and raises on a declared adapter that fails its
    mandatory round-trip proof. The final path is replaced only after a second
    restore/probe succeeds.
    """

    del probe_inputs  # Adapters own deterministic canonical probes in schema v1.
    _load_builtin_adapters()
    from pyimgano.exporting import get_export_adapter, get_state_codec

    model_name = str(config.model.name)
    try:
        adapter = get_export_adapter(model_name)
    except KeyError:
        return None

    codec_id = str(getattr(adapter, "state_codec_id", "") or "").strip()
    if not codec_id:
        raise CheckpointCertificationError(
            f"Export adapter {adapter.adapter_id!r} does not declare a safe state codec."
        )
    codec = get_state_codec(codec_id)
    original_path = Path(checkpoint_path)
    if not original_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {original_path}")

    from pyimgano.inference.runtime_wrappers import unwrap_runtime_detector
    from pyimgano.training.checkpointing import build_checkpoint_contract

    original = unwrap_runtime_detector(detector)
    context = {
        "phase": "post_training_checkpoint_certification",
        "model_name": model_name,
        "model_kwargs": _resolved_model_kwargs(config),
    }
    restored = _fresh_restore_detector(
        adapter,
        original=original,
        config=config,
        context=context,
    )
    state = dict(codec.encode(original))
    codec.validate_state(state)
    codec.decode(restored, state)
    first_probe = _verify_roundtrip(
        adapter,
        original=original,
        restored=restored,
        codec=codec,
        context=context,
    )

    fingerprint_payload = _checkpoint_fingerprint_payload(
        adapter,
        original=original,
        context=context,
    )
    contract = build_checkpoint_contract(
        original_path,
        codec_id=codec_id,
        codec_version=int(codec.codec_version),
        adapter_id=str(adapter.adapter_id),
        adapter_version=int(adapter.adapter_version),
        model_config=fingerprint_payload,
        state_schema_version=int(codec.state_schema_version),
        roundtrip_verified=True,
        roundtrip=first_probe,
    )

    from pyimgano.exporting import load_fitted_state, save_fitted_state

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{original_path.name}.certified-",
            suffix=".tmp",
            dir=original_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        temporary.unlink()
        save_fitted_state(
            original,
            temporary,
            model_name=model_name,
            checkpoint_contract=contract,
            codec_id=codec_id,
        )
        final_restored = _fresh_restore_detector(
            adapter,
            original=original,
            config=config,
            context=context,
        )
        load_fitted_state(final_restored, temporary, expected_model_name=model_name)
        final_probe = _verify_roundtrip(
            adapter,
            original=original,
            restored=final_restored,
            codec=codec,
            context=context,
        )
        if final_probe.get("passed") is not True:
            raise CheckpointCertificationError("Final persisted checkpoint parity failed.")
        os.replace(temporary, original_path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    from pyimgano.training.checkpointing import inspect_checkpoint_contract

    final_contract = inspect_checkpoint_contract(original_path)
    if not final_contract.strict_exportable:
        raise CheckpointCertificationError(
            "Certified checkpoint did not retain a complete strict export contract."
        )
    return final_contract


__all__ = [
    "CheckpointCertificationError",
    "certify_checkpoint_for_export",
]
