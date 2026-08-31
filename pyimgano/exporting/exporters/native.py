from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pyimgano.artifacts.compatibility import current_platform_tag
from pyimgano.exporting.protocols import ExportAdapterProtocol
from pyimgano.exporting.registry import get_export_adapter, get_export_capability
from pyimgano.exporting.state_codec import get_state_codec, save_fitted_state
from pyimgano.exporting.types import (
    ArtifactFormat,
    ExportLayout,
    ExportStatus,
    ExportTargetKind,
    NativeExportContext,
    NativeExportResult,
)
from pyimgano.exporting.writer import ArtifactWriter


class NativeExportError(RuntimeError):
    pass


def _validate_adapter_contract(
    adapter: ExportAdapterProtocol,
    context: NativeExportContext,
) -> str:
    contract = context.checkpoint_contract
    if not contract.strict_exportable:
        raise NativeExportError(
            "Native artifact export requires a complete checkpoint with recorded "
            "round-trip evidence; loadability cannot upgrade unknown completeness."
        )
    if contract.adapter_id != str(adapter.adapter_id) or int(contract.adapter_version or 0) != int(
        adapter.adapter_version
    ):
        raise NativeExportError(
            "Checkpoint adapter identity does not match the selected export adapter."
        )
    codec_id = str(getattr(adapter, "state_codec_id", "") or "").strip()
    if not codec_id:
        raise NativeExportError("Native export adapter does not declare a safe state codec.")
    if contract.codec_id != codec_id:
        raise NativeExportError("Checkpoint codec identity does not match the export adapter.")
    codec = get_state_codec(codec_id, contract.codec_version)
    if int(codec.codec_version) != int(contract.codec_version or 0):
        raise NativeExportError("Checkpoint codec version does not match the registered codec.")
    if int(codec.state_schema_version) != int(contract.state_schema_version or 0):
        raise NativeExportError("Checkpoint state schema does not match the registered codec.")
    validate = getattr(adapter, "validate_checkpoint_contract", None)
    if callable(validate):
        validate(contract, context=context)
    return codec_id


def _compatibility_payload(
    context: NativeExportContext,
    adapter: ExportAdapterProtocol,
    *,
    codec_id: str,
) -> dict[str, Any]:
    codec = get_state_codec(codec_id, context.checkpoint_contract.codec_version)
    raw_runtime_versions = getattr(adapter, "native_runtime_versions", {})
    if not isinstance(raw_runtime_versions, Mapping):
        raise NativeExportError("Adapter native_runtime_versions must be a mapping.")
    runtime_versions = dict(raw_runtime_versions)
    payload: dict[str, Any] = {
        "pyimgano": ">=0.10,<0.11",
        "python": ">=3.9,<3.13",
        "platforms": [current_platform_tag()],
        "runtime_versions": runtime_versions,
        "adapter": {"id": str(adapter.adapter_id), "version": int(adapter.adapter_version)},
        "codecs": [{"id": codec_id, "version": int(codec.codec_version)}],
    }
    payload.update(dict(context.compatibility))
    # These bindings are executable authority and cannot be overridden by caller
    # compatibility hints.
    payload["adapter"] = {
        "id": str(adapter.adapter_id),
        "version": int(adapter.adapter_version),
    }
    payload["codecs"] = [{"id": codec_id, "version": int(codec.codec_version)}]
    payload["platforms"] = [current_platform_tag()]
    payload["runtime_versions"] = runtime_versions
    return payload


def _runtime_payload(
    adapter: ExportAdapterProtocol,
    context: NativeExportContext,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "backend": "pyimgano",
        "allowed_providers": [{"name": "CPU", "options": {}}],
        "verified_providers": [{"name": "CPU", "options": {}}],
        "entrypoint": "state/detector.pyim",
    }
    build_runtime = getattr(adapter, "build_runtime_spec", None)
    if callable(build_runtime):
        runtime.update(dict(build_runtime(format=ArtifactFormat.NATIVE, context=context) or {}))
    if runtime.get("backend") != "pyimgano":
        raise NativeExportError("Native detector artifacts require runtime.backend='pyimgano'.")
    runtime["entrypoint"] = "state/detector.pyim"
    return runtime


def export_native(
    detector: Any,
    *,
    context: NativeExportContext,
    out: str | Path,
    adapter: ExportAdapterProtocol | None = None,
    overwrite: bool = False,
) -> NativeExportResult:
    """Export one restored fitted detector as a safe native artifact."""

    selected = adapter if adapter is not None else get_export_adapter(context.model_name)
    capability = get_export_capability(
        context.model_name,
        ArtifactFormat.NATIVE,
        context=context,
    )
    if adapter is not None:
        effective_fn = getattr(selected, "effective_capability", None)
        capability = (
            effective_fn(ArtifactFormat.NATIVE, context=context)
            if callable(effective_fn)
            else selected.declared_capability(ArtifactFormat.NATIVE)
        )
    if (
        capability.status is not ExportStatus.SUPPORTED
        or capability.target_kind is not ExportTargetKind.ARTIFACT
        or capability.layout is not ExportLayout.NATIVE_DETECTOR
    ):
        raise NativeExportError(
            "Model does not have effective native-detector export support: "
            f"model={context.model_name!r}, reason={capability.reason_code!r}."
        )

    codec_id = _validate_adapter_contract(selected, context)
    out_path = Path(out)
    with ArtifactWriter(out_path, overwrite=overwrite) as writer:
        state_path = writer.path_for("state/detector.pyim")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_state(
            detector,
            state_path,
            model_name=context.model_name,
            checkpoint_contract=context.checkpoint_contract,
            codec_id=codec_id,
        )
        state_component = writer.component_metadata(
            "state/detector.pyim",
            component_id="trained-state",
            role="trained_state",
            format="pyimgano-state",
            serialization="safe-data",
        )

        verification_payload = dict(context.verification)
        verification_level = str(verification_payload.pop("level", "reference_parity"))
        verification_payload.setdefault("checkpoint_roundtrip_verified", True)
        verification_payload.setdefault(
            "source_checkpoint_sha256", context.checkpoint_contract.sha256
        )
        verification_payload.setdefault(
            "checkpoint_roundtrip", dict(context.checkpoint_contract.roundtrip)
        )
        verification_path = writer.write_json(
            "verification/parity.json",
            verification_payload,
        )
        verification_report = {
            "path": "verification/parity.json",
            "size_bytes": int(verification_path.stat().st_size),
            "sha256": writer.component_metadata(
                "verification/parity.json",
                component_id="verification",
                role="verification",
                format="json",
                serialization="safe-data",
            )["sha256"],
        }

        model_payload: dict[str, Any] = {
            "registry_name": str(context.model_name),
            "constructor_kwargs": dict(context.model_kwargs),
        }
        if context.category is not None:
            model_payload["category"] = str(context.category)
        manifest_payload: dict[str, Any] = {
            "schema_family": "pyimgano-artifact",
            "schema_version": 1,
            "layout": ExportLayout.NATIVE_DETECTOR.value,
            "model": model_payload,
            "runtime": _runtime_payload(selected, context),
            "input_contract": dict(context.input_contract),
            "output_contract": dict(context.output_contract),
            "components": [state_component],
            "policy_ref": {"path": "infer_config.json"},
            "compatibility": _compatibility_payload(context, selected, codec_id=codec_id),
            "verification": {
                "level": verification_level,
                "reference_backend": "pyimgano",
                "report": verification_report,
            },
        }
        manifest_path, manifest = writer.finalize(
            manifest_payload,
            policy=dict(context.policy),
        )

    return NativeExportResult(
        artifact_root=out_path,
        manifest_path=manifest_path,
        state_path=out_path / "state" / "detector.pyim",
        policy_path=out_path / "infer_config.json",
        manifest=manifest,
    )


__all__ = ["NativeExportError", "export_native"]
