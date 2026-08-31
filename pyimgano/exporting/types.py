from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class ArtifactFormat(_StringEnum):
    NATIVE = "native"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    OPENVINO = "openvino"


class ExportStatus(_StringEnum):
    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


class ExportLayout(_StringEnum):
    NATIVE_DETECTOR = "native_detector"
    SINGLE_GRAPH = "single_graph"
    COMPOSITE = "composite"


class ExportTargetKind(_StringEnum):
    ARTIFACT = "artifact"
    EXTERNAL_CHECKPOINT = "external_checkpoint"


class CapabilityAvailability(_StringEnum):
    AVAILABLE = "available"
    MISSING_EXTRA = "missing_extra"


class CheckpointCompleteness(_StringEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    FAILED = "failed"


class SerializationKind(_StringEnum):
    SAFE_DATA = "safe-data"
    EXECUTABLE_TRUST_REQUIRED = "executable-trust-required"


@dataclass(frozen=True)
class ExportCapability:
    """One orthogonal trained-export capability cell.

    Runtime consumption (for example an ONNX embedding extractor) is deliberately
    not represented by this object.  A cell describes exporting a *fitted detector*
    for one requested artifact format.
    """

    format: ArtifactFormat
    status: ExportStatus
    target_kind: ExportTargetKind = ExportTargetKind.ARTIFACT
    layout: ExportLayout | None = None
    availability: CapabilityAvailability = CapabilityAvailability.AVAILABLE
    adapter_id: str | None = None
    adapter_version: int | None = None
    conditions: tuple[str, ...] = ()
    reason_code: str | None = None
    remediation: str | None = None

    def __post_init__(self) -> None:
        if self.status is ExportStatus.UNSUPPORTED and not self.reason_code:
            raise ValueError("Unsupported export capabilities require reason_code.")
        if (
            self.status is ExportStatus.SUPPORTED
            and self.target_kind is ExportTargetKind.ARTIFACT
            and self.layout is None
        ):
            raise ValueError("Supported export capabilities require a layout.")
        if self.target_kind is ExportTargetKind.EXTERNAL_CHECKPOINT and self.layout is not None:
            raise ValueError("External checkpoint targets do not use an artifact layout.")
        if self.adapter_version is not None and int(self.adapter_version) < 1:
            raise ValueError("adapter_version must be a positive integer.")

    @property
    def supported(self) -> bool:
        return bool(
            self.status is ExportStatus.SUPPORTED
            and self.availability is CapabilityAvailability.AVAILABLE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": str(self.format),
            "status": str(self.status),
            "target_kind": str(self.target_kind),
            "layout": (str(self.layout) if self.layout is not None else None),
            "availability": str(self.availability),
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "conditions": list(self.conditions),
            "reason_code": self.reason_code,
            "remediation": self.remediation,
        }

    @classmethod
    def unsupported(
        cls,
        format: ArtifactFormat | str,
        *,
        reason_code: str,
        remediation: str | None = None,
        availability: CapabilityAvailability = CapabilityAvailability.AVAILABLE,
        conditions: tuple[str, ...] = (),
    ) -> "ExportCapability":
        return cls(
            format=ArtifactFormat(str(format)),
            status=ExportStatus.UNSUPPORTED,
            availability=availability,
            conditions=tuple(conditions),
            reason_code=str(reason_code),
            remediation=(str(remediation) if remediation is not None else None),
        )


@dataclass(frozen=True)
class CheckpointContract:
    """Persisted evidence about a fitted checkpoint.

    ``from_mapping`` intentionally defaults missing completeness to ``unknown``.
    Merely loading a legacy checkpoint must never promote this field.
    """

    completeness: CheckpointCompleteness = CheckpointCompleteness.UNKNOWN
    codec_id: str | None = None
    codec_version: int | None = None
    adapter_id: str | None = None
    adapter_version: int | None = None
    model_config_fingerprint: str | None = None
    state_schema_version: int | None = None
    serialization: SerializationKind = SerializationKind.SAFE_DATA
    requires_trust: bool = False
    size_bytes: int | None = None
    sha256: str | None = None
    roundtrip_verified: bool = False
    roundtrip: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.codec_version is not None and int(self.codec_version) < 1:
            raise ValueError("codec_version must be a positive integer.")
        if self.adapter_version is not None and int(self.adapter_version) < 1:
            raise ValueError("adapter_version must be a positive integer.")
        if self.state_schema_version is not None and int(self.state_schema_version) < 1:
            raise ValueError("state_schema_version must be a positive integer.")
        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("size_bytes must not be negative.")
        if self.sha256 is not None:
            digest = str(self.sha256).strip().lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("sha256 must be a 64-character hexadecimal digest.")
            object.__setattr__(self, "sha256", digest)
        if (
            self.requires_trust
            and self.serialization is not SerializationKind.EXECUTABLE_TRUST_REQUIRED
        ):
            raise ValueError(
                "requires_trust=True requires executable-trust-required serialization."
            )

    @property
    def strict_exportable(self) -> bool:
        """Whether the metadata carries explicit complete round-trip evidence."""

        return bool(
            self.completeness is CheckpointCompleteness.COMPLETE
            and self.roundtrip_verified
            and self.codec_id
            and self.codec_version is not None
            and self.adapter_id
            and self.adapter_version is not None
            and self.model_config_fingerprint
            and self.state_schema_version is not None
            and self.sha256
            and self.size_bytes is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": str(self.completeness),
            "codec_id": self.codec_id,
            "codec_version": self.codec_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "model_config_fingerprint": self.model_config_fingerprint,
            "state_schema_version": self.state_schema_version,
            "serialization": str(self.serialization),
            "requires_trust": bool(self.requires_trust),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "roundtrip_verified": bool(self.roundtrip_verified),
            "roundtrip": dict(self.roundtrip),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CheckpointContract":
        data = dict(payload or {})
        completeness_raw = data.get("completeness", CheckpointCompleteness.UNKNOWN.value)
        serialization_raw = data.get("serialization", SerializationKind.SAFE_DATA.value)
        roundtrip = data.get("roundtrip", {})
        if not isinstance(roundtrip, Mapping):
            raise ValueError("checkpoint roundtrip metadata must be a mapping.")
        return cls(
            completeness=CheckpointCompleteness(str(completeness_raw)),
            codec_id=(str(data["codec_id"]) if data.get("codec_id") is not None else None),
            codec_version=(
                int(data["codec_version"]) if data.get("codec_version") is not None else None
            ),
            adapter_id=(str(data["adapter_id"]) if data.get("adapter_id") is not None else None),
            adapter_version=(
                int(data["adapter_version"]) if data.get("adapter_version") is not None else None
            ),
            model_config_fingerprint=(
                str(data["model_config_fingerprint"])
                if data.get("model_config_fingerprint") is not None
                else None
            ),
            state_schema_version=(
                int(data["state_schema_version"])
                if data.get("state_schema_version") is not None
                else None
            ),
            serialization=SerializationKind(str(serialization_raw)),
            requires_trust=bool(data.get("requires_trust", False)),
            size_bytes=(int(data["size_bytes"]) if data.get("size_bytes") is not None else None),
            sha256=(str(data["sha256"]) if data.get("sha256") is not None else None),
            roundtrip_verified=bool(data.get("roundtrip_verified", False)),
            roundtrip=dict(roundtrip),
            failure_reason=(
                str(data["failure_reason"]) if data.get("failure_reason") is not None else None
            ),
        )


@dataclass(frozen=True)
class ProbeSpec:
    inputs: tuple[Any, ...]
    expected_outputs: tuple[str, ...] = ("score",)
    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-4


@dataclass(frozen=True)
class GraphExportSpec:
    inference_module: Any
    example_inputs: tuple[Any, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    output_semantics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentExportSpec:
    components: tuple[Mapping[str, Any], ...]
    bindings: Mapping[str, Any]


@dataclass(frozen=True)
class NativeExportContext:
    model_name: str
    model_kwargs: Mapping[str, Any]
    policy: Mapping[str, Any]
    checkpoint_contract: CheckpointContract
    category: str | None = None
    verification: Mapping[str, Any] = field(default_factory=dict)
    input_contract: Mapping[str, Any] = field(
        default_factory=lambda: {
            "kind": "image_batch",
            "dtype": "uint8",
            "layout": "HWC",
        }
    )
    output_contract: Mapping[str, Any] = field(
        default_factory=lambda: {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        }
    )
    compatibility: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeExportResult:
    artifact_root: Path
    manifest_path: Path
    state_path: Path
    policy_path: Path
    manifest: Mapping[str, Any]


__all__ = [
    "ArtifactFormat",
    "CapabilityAvailability",
    "CheckpointCompleteness",
    "CheckpointContract",
    "ComponentExportSpec",
    "ExportCapability",
    "ExportLayout",
    "ExportStatus",
    "ExportTargetKind",
    "GraphExportSpec",
    "NativeExportContext",
    "NativeExportResult",
    "ProbeSpec",
    "SerializationKind",
]
