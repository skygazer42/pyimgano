from __future__ import annotations

from dataclasses import dataclass

import pytest

from pyimgano.exporting.registry import (
    ExportAdapterRegistrationError,
    ExportAdapterRegistry,
)
from pyimgano.exporting.types import (
    ArtifactFormat,
    ExportCapability,
    ExportLayout,
    ExportStatus,
)


@dataclass
class _Adapter:
    adapter_id: str = "test.reference"
    adapter_version: int = 1
    model_names: tuple[str, ...] = ("test_model",)
    state_codec_id: str = "test.state"

    def declared_capability(self, format: ArtifactFormat) -> ExportCapability:
        if format is ArtifactFormat.NATIVE:
            return ExportCapability(
                format=format,
                status=ExportStatus.SUPPORTED,
                layout=ExportLayout.NATIVE_DETECTOR,
            )
        return ExportCapability.unsupported(format, reason_code="format_not_certified")

    def effective_capability(self, format: ArtifactFormat, *, context):  # noqa: ANN001
        if context.get("checkpoint_complete") is False:
            return ExportCapability.unsupported(format, reason_code="checkpoint_incomplete")
        return self.declared_capability(format)


def test_registry_missing_adapter_is_explicitly_unsupported() -> None:
    registry = ExportAdapterRegistry()

    capability = registry.capability("vision_onnx_ecod", ArtifactFormat.ONNX)

    assert capability.status is ExportStatus.UNSUPPORTED
    assert capability.reason_code == "no_export_adapter"
    assert capability.adapter_id is None


def test_registry_separates_declared_and_effective_capability() -> None:
    registry = ExportAdapterRegistry()
    registry.register(_Adapter(), aliases=("test_alias",))

    declared = registry.capability("test_alias", "native")
    effective = registry.capability(
        "test_model",
        "native",
        context={"checkpoint_complete": False},
    )

    assert declared.supported is True
    assert declared.adapter_id == "test.reference"
    assert effective.supported is False
    assert effective.reason_code == "checkpoint_incomplete"


def test_registry_rejects_conflicting_aliases_and_adapter_ids() -> None:
    registry = ExportAdapterRegistry()
    registry.register(_Adapter(), aliases=("test_alias",))

    with pytest.raises(ExportAdapterRegistrationError, match="already registered"):
        registry.register(
            _Adapter(adapter_id="other", model_names=("other_model",)),
            aliases=("test_alias",),
        )
    with pytest.raises(ExportAdapterRegistrationError, match="id already"):
        registry.register(_Adapter(model_names=("second_model",)))


def test_support_table_contains_every_format_and_unsupported_reason() -> None:
    registry = ExportAdapterRegistry()
    registry.register(_Adapter())

    table = registry.support_table(model_names=["test_model", "missing"])

    assert set(table["test_model"]) == {str(value) for value in ArtifactFormat}
    assert table["test_model"]["native"]["status"] == "supported"
    assert table["test_model"]["onnx"]["reason_code"] == "format_not_certified"
    assert table["missing"]["native"]["reason_code"] == "no_export_adapter"


def test_onnx_consumption_tag_does_not_imply_trained_export() -> None:
    from pyimgano.models.capabilities import (
        compute_runtime_consumption_capabilities,
        compute_trained_export_capabilities,
    )

    @dataclass
    class _Entry:
        name: str = "unregistered_onnx_consumer"
        constructor: object = object
        tags: tuple[str, ...] = ("vision", "onnx")
        metadata: dict[str, object] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            self.metadata = {"weights_source": "local-exported-onnx"}

    entry = _Entry()
    assert compute_runtime_consumption_capabilities(entry)["onnx"] is True
    assert compute_trained_export_capabilities(entry)["onnx"]["status"] == "unsupported"


def test_builtin_ecod_embedding_models_use_one_explicit_composite_adapter() -> None:
    from pyimgano.exporting import get_export_adapter, get_export_capability

    onnx_adapter = get_export_adapter("vision_onnx_ecod")
    torchscript_adapter = get_export_adapter("vision_torchscript_ecod")

    assert onnx_adapter is torchscript_adapter
    assert onnx_adapter.adapter_id == "pyimgano.embedding-core-ecod"
    onnx = get_export_capability("vision_onnx_ecod", ArtifactFormat.ONNX)
    torchscript = get_export_capability("vision_torchscript_ecod", ArtifactFormat.TORCHSCRIPT)
    assert onnx.status is ExportStatus.CONDITIONAL
    assert onnx.layout is ExportLayout.COMPOSITE
    assert torchscript.status is ExportStatus.CONDITIONAL
    assert torchscript.layout is ExportLayout.COMPOSITE
    assert (
        get_export_capability("vision_onnx_ecod", ArtifactFormat.TORCHSCRIPT).status
        is ExportStatus.UNSUPPORTED
    )
    assert (
        get_export_capability("vision_onnx_ecod", ArtifactFormat.TORCHSCRIPT).reason_code
        == "extractor_format_mismatch"
    )

    table = {
        model: {format.value: get_export_capability(model, format) for format in ArtifactFormat}
        for model in ("vision_onnx_ecod", "vision_torchscript_ecod")
    }
    assert [
        name
        for name, capability in table["vision_onnx_ecod"].items()
        if capability.status is ExportStatus.CONDITIONAL
    ] == ["onnx"]
    assert [
        name
        for name, capability in table["vision_torchscript_ecod"].items()
        if capability.status is ExportStatus.CONDITIONAL
    ] == ["torchscript"]
    for model, cells in table.items():
        expected = "onnx" if "onnx" in model else "torchscript"
        assert all(
            capability.reason_code == "extractor_format_mismatch"
            for name, capability in cells.items()
            if name != expected
        )
