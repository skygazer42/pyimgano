from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pyimgano.exporting.types import (
    ArtifactFormat,
    CheckpointContract,
    ComponentExportSpec,
    ExportCapability,
    GraphExportSpec,
    NativeExportContext,
    ProbeSpec,
)


@runtime_checkable
class CheckpointRoundTripProtocol(Protocol):
    """Adapter evidence needed to persist raw score/map computation state."""

    adapter_id: str
    adapter_version: int
    state_codec_id: str

    def validate_checkpoint_contract(
        self,
        contract: CheckpointContract,
        *,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> None: ...

    def restore_state(self, detector: Any, checkpoint: str | Path) -> None: ...

    def build_probe_spec(
        self,
        detector: Any,
        *,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> ProbeSpec: ...


@runtime_checkable
class GraphExportProtocol(Protocol):
    def build_graph_export_spec(
        self,
        detector: Any,
        *,
        format: ArtifactFormat,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> GraphExportSpec: ...


@runtime_checkable
class ComponentExportProtocol(Protocol):
    def export_components(
        self,
        detector: Any,
        *,
        format: ArtifactFormat,
        output_dir: Path,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> ComponentExportSpec: ...


@runtime_checkable
class ExportAdapterProtocol(Protocol):
    adapter_id: str
    adapter_version: int
    model_names: Sequence[str]
    state_codec_id: str | None

    def inspect_source(self, source: Any) -> Mapping[str, Any]: ...

    def declared_capability(self, format: ArtifactFormat) -> ExportCapability: ...

    def effective_capability(
        self,
        format: ArtifactFormat,
        *,
        context: Mapping[str, Any] | NativeExportContext,
    ) -> ExportCapability: ...

    def validate_checkpoint_contract(
        self,
        contract: CheckpointContract,
        *,
        context: NativeExportContext | Mapping[str, Any] | None = None,
    ) -> None: ...

    def build_runtime_spec(
        self,
        *,
        format: ArtifactFormat,
        context: NativeExportContext | Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


__all__ = [
    "CheckpointRoundTripProtocol",
    "ComponentExportProtocol",
    "ExportAdapterProtocol",
    "GraphExportProtocol",
]
