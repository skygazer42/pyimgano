from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from pyimgano.exporting.exporters.native import NativeExportError, export_native
from pyimgano.exporting.registry import ExportAdapterRegistry
from pyimgano.exporting.state_codec import (
    MappingStateCodec,
    StateCodecRegistry,
    StateField,
    load_fitted_state,
)
from pyimgano.exporting.types import (
    ArtifactFormat,
    CheckpointCompleteness,
    CheckpointContract,
    ExportCapability,
    ExportLayout,
    ExportStatus,
    NativeExportContext,
)


class _Detector:
    def __init__(self, weights, threshold: float) -> None:  # noqa: ANN001
        self.weights = np.asarray(weights, dtype=np.float32)
        self.threshold_ = float(threshold)


class _Adapter:
    adapter_id = "test.reference"
    adapter_version = 1
    model_names = ("test_native_model",)
    state_codec_id = "test.weights"

    def declared_capability(self, format: ArtifactFormat) -> ExportCapability:
        if format is ArtifactFormat.NATIVE:
            return ExportCapability(
                format=format,
                status=ExportStatus.SUPPORTED,
                layout=ExportLayout.NATIVE_DETECTOR,
            )
        return ExportCapability.unsupported(format, reason_code="not_certified")

    def effective_capability(self, format: ArtifactFormat, *, context):  # noqa: ANN001
        return self.declared_capability(format)

    def validate_checkpoint_contract(self, contract, *, context=None) -> None:  # noqa: ANN001
        if not contract.strict_exportable:
            raise ValueError("checkpoint incomplete")

    def build_runtime_spec(self, *, format: ArtifactFormat, context):  # noqa: ANN001
        return {}


def _contract() -> CheckpointContract:
    return CheckpointContract(
        completeness=CheckpointCompleteness.COMPLETE,
        codec_id="test.weights",
        codec_version=1,
        adapter_id="test.reference",
        adapter_version=1,
        model_config_fingerprint="sha256:" + "a" * 64,
        state_schema_version=1,
        size_bytes=6,
        sha256=hashlib.sha256(b"source").hexdigest(),
        roundtrip_verified=True,
        roundtrip={"probe": "passed"},
    )


def _policy() -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "model": {
            "registry_name": "test_native_model",
            "category": "bottle",
            "constructor_kwargs": {"device": "cpu"},
        },
        "postprocess": {
            "image_threshold": {
                "threshold": 0.7,
                "score_order": "higher_is_more_anomalous",
            }
        },
    }


@pytest.fixture
def isolated_registries(monkeypatch):  # noqa: ANN001
    import pyimgano.exporting.registry as adapter_module
    import pyimgano.exporting.state_codec as codec_module

    adapter_registry = ExportAdapterRegistry()
    adapter_registry.register(_Adapter())
    codec_registry = StateCodecRegistry()
    codec_registry.register(
        MappingStateCodec(
            codec_id="test.weights",
            codec_version=1,
            state_schema_version=1,
            model_names=("test_native_model",),
            fields=(StateField("weights", dtypes=("float32",), ranks=(1,)),),
        )
    )
    monkeypatch.setattr(adapter_module, "EXPORT_ADAPTER_REGISTRY", adapter_registry)
    monkeypatch.setattr(codec_module, "STATE_CODEC_REGISTRY", codec_registry)
    return adapter_registry, codec_registry


def _context(contract: CheckpointContract) -> NativeExportContext:
    return NativeExportContext(
        model_name="test_native_model",
        model_kwargs={"device": "cpu"},
        category="bottle",
        policy=_policy(),
        checkpoint_contract=contract,
        verification={"level": "reference_parity", "probe": "passed"},
        compatibility={"platforms": ["linux-x86_64"]},
    )


def test_native_export_writes_valid_relocatable_safe_artifact(
    tmp_path,
    isolated_registries,
    monkeypatch,
) -> None:
    import pyimgano.exporting.exporters.native as native_module
    from pyimgano.artifacts import load_artifact_manifest, verify_artifact_files

    monkeypatch.setattr(native_module, "current_platform_tag", lambda: "macos-arm64")

    out = tmp_path / "native"
    detector = _Detector([1.0, 2.0], threshold=0.7)

    result = export_native(detector, context=_context(_contract()), out=out)

    manifest = load_artifact_manifest(out)
    verify_artifact_files(out, manifest)
    assert result.manifest_path == out / "artifact_manifest.json"
    assert manifest["layout"] == "native_detector"
    assert manifest["components"][0]["role"] == "trained_state"
    assert manifest["compatibility"]["platforms"] == ["macos-arm64"]
    assert manifest["compatibility"]["runtime_versions"] == {}
    restored = _Detector([0.0, 0.0], threshold=-5.0)
    load_fitted_state(
        restored,
        result.state_path,
        expected_model_name="test_native_model",
    )
    np.testing.assert_allclose(restored.weights, detector.weights)
    assert restored.threshold_ == -5.0


def test_native_export_rejects_unknown_checkpoint_without_partial_output(
    tmp_path,
    isolated_registries,
) -> None:
    unknown = replace(
        _contract(),
        completeness=CheckpointCompleteness.UNKNOWN,
        roundtrip_verified=False,
    )
    out = tmp_path / "native"

    with pytest.raises(NativeExportError, match="loadability cannot upgrade"):
        export_native(_Detector([1.0], threshold=0.2), context=_context(unknown), out=out)

    assert not out.exists()


def test_native_export_refuses_existing_output_without_mutating_it(
    tmp_path,
    isolated_registries,
) -> None:
    out = tmp_path / "native"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_native(_Detector([1.0], threshold=0.2), context=_context(_contract()), out=out)

    assert marker.read_text(encoding="utf-8") == "keep"
