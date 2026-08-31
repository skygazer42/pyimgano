from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyimgano.exporting import (
    ArtifactFormat,
    CapabilityAvailability,
    ExportLayout,
    ExportStatus,
    NativeExportContext,
    get_export_adapter,
    get_export_capability,
)


@dataclass
class _FittedCase:
    detector: Any
    adapter: Any
    context: NativeExportContext
    inputs: list[np.ndarray]
    reference: dict[str, np.ndarray]


@pytest.fixture(scope="module")
def fitted_case(tmp_path_factory) -> _FittedCase:  # noqa: ANN001
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from pyimgano.models.ae import OptimizedAEDetector
    from pyimgano.services.checkpoint_certification_service import (
        certify_checkpoint_for_export,
    )
    from pyimgano.training.checkpointing import save_checkpoint

    root = tmp_path_factory.mktemp("trained-ae-export")
    train_images = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (0, 80, 160, 240)]
    detector = OptimizedAEDetector(
        tiny=True,
        image_size=16,
        epochs=1,
        batch_size=2,
        device="cpu",
        verbose=0,
        random_state=7,
    )
    detector.fit(train_images)
    adapter = get_export_adapter("ae_resnet_unet")
    checkpoint = save_checkpoint(detector, root / "model.pt")

    kwargs = {
        "contamination": 0.1,
        "tiny": True,
        "image_size": 16,
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
        "verbose": 0,
        "random_state": 7,
    }
    config = SimpleNamespace(
        seed=7,
        model=SimpleNamespace(
            name="ae_resnet_unet",
            model_kwargs={
                key: value
                for key, value in kwargs.items()
                if key not in {"contamination", "device", "random_state"}
            },
            device="cpu",
            contamination=0.1,
            pretrained=False,
            preset=None,
        ),
    )
    contract = certify_checkpoint_for_export(detector, checkpoint, config=config)
    assert contract is not None and contract.strict_exportable

    probe = adapter.build_probe_spec(detector, context={"model_kwargs": kwargs})
    restored = OptimizedAEDetector(**kwargs)
    adapter.restore_state(restored, checkpoint)
    roundtrip = adapter.verify_roundtrip(detector, restored, probe)
    assert roundtrip["passed"] is True
    threshold = float(detector.threshold_)
    policy = {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "model": {
            "registry_name": "ae_resnet_unet",
            "category": "reference",
            "constructor_kwargs": kwargs,
        },
        "postprocess": {
            "image_threshold": {
                "threshold": threshold,
                "score_order": "higher_is_more_anomalous",
            }
        },
    }
    context = NativeExportContext(
        model_name="ae_resnet_unet",
        model_kwargs=kwargs,
        category="reference",
        policy=policy,
        checkpoint_contract=contract,
        verification={"level": "reference_parity", "source": "focused_test"},
    )
    return _FittedCase(
        detector=detector,
        adapter=adapter,
        context=context,
        inputs=list(probe.inputs),
        reference={
            name: np.asarray(value, dtype=np.float32)
            for name, value in adapter.evaluate_probe(detector, probe).items()
        },
    )


def test_builtin_adapter_registration_and_declared_capability_matrix() -> None:
    adapter = get_export_adapter("ae_resnet_unet")

    assert adapter.adapter_id == "pyimgano.ae-resnet-unet"
    assert adapter.native_runtime_versions == {"torch": ">=1.9"}
    native = get_export_capability("ae_resnet_unet", ArtifactFormat.NATIVE)
    assert native.status is ExportStatus.SUPPORTED
    assert native.layout is ExportLayout.NATIVE_DETECTOR
    for format in (
        ArtifactFormat.ONNX,
        ArtifactFormat.TORCHSCRIPT,
        ArtifactFormat.OPENVINO,
    ):
        capability = get_export_capability("ae_resnet_unet", format)
        assert capability.status is ExportStatus.CONDITIONAL
        assert capability.layout is ExportLayout.SINGLE_GRAPH
        assert capability.reason_code == "requires_concrete_export_context"


def test_optional_dependency_capability_fails_closed(monkeypatch) -> None:  # noqa: ANN001
    import pyimgano.exporting.adapters.autoencoder as module

    adapter = get_export_adapter("ae_resnet_unet")
    monkeypatch.setattr(
        module,
        "_module_available",
        lambda name: name not in {"onnx", "onnxruntime"},
    )

    capability = adapter.effective_capability(
        ArtifactFormat.ONNX,
        context={"phase": "pre_training", "model_kwargs": {"image_size": 16}},
    )

    assert capability.status is ExportStatus.UNSUPPORTED
    assert capability.availability is CapabilityAvailability.MISSING_EXTRA
    assert capability.reason_code == "missing_export_dependency"
    assert "pyimgano[onnx-export]" in str(capability.remediation)


def test_codec_roundtrip_restores_weights_but_not_operating_threshold(
    fitted_case: _FittedCase,
    tmp_path: Path,
) -> None:
    from pyimgano.exporting import load_fitted_state, save_fitted_state
    from pyimgano.models.ae import OptimizedAEDetector

    state_path = save_fitted_state(
        fitted_case.detector,
        tmp_path / "detector.pyim",
        model_name="ae_resnet_unet",
        checkpoint_contract=fitted_case.context.checkpoint_contract,
    )
    restored = OptimizedAEDetector(**dict(fitted_case.context.model_kwargs))
    restored.threshold_ = -123.0

    load_fitted_state(restored, state_path, expected_model_name="ae_resnet_unet")
    report = fitted_case.adapter.verify_roundtrip(fitted_case.detector, restored)

    assert report["passed"] is True
    assert restored.threshold_ == -123.0


@pytest.mark.parametrize(
    ("format", "component_format", "backend"),
    [
        (ArtifactFormat.ONNX, "onnx", "onnxruntime"),
        (ArtifactFormat.TORCHSCRIPT, "torchscript", "torchscript"),
    ],
)
def test_fitted_graph_export_manifest_public_load_and_reference_parity(
    format: ArtifactFormat,
    component_format: str,
    backend: str,
    fitted_case: _FittedCase,
    tmp_path: Path,
) -> None:
    if format is ArtifactFormat.ONNX:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    else:
        pytest.importorskip("torch")
    from pyimgano.artifacts import load_artifact_manifest, verify_artifact_files
    from pyimgano.artifacts.compatibility import current_platform_tag
    from pyimgano.inference import load_artifact

    out = tmp_path / format.value
    result = fitted_case.adapter.export_artifact(
        fitted_case.detector,
        format=format,
        context=replace(
            fitted_case.context,
            compatibility={"platforms": ["windows-x86_64"]},
        ),
        out=out,
    )
    manifest = load_artifact_manifest(out)
    verify_artifact_files(out, manifest)

    assert result.manifest_path == out / "artifact_manifest.json"
    assert manifest["layout"] == "single_graph"
    assert manifest["runtime"]["backend"] == backend
    assert manifest["components"][0]["format"] == component_format
    assert manifest["verification"]["level"] == "reference_parity"
    assert manifest["compatibility"]["platforms"] == [current_platform_tag()]
    assert manifest["output_contract"]["score"]["score_order"] == ("higher_is_more_anomalous")
    assert manifest["input_contract"]["normalize"] == {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    runtime = load_artifact(
        out,
        format=component_format,
        backend=backend,
        trust_checkpoint=format is ArtifactFormat.TORCHSCRIPT,
    )
    try:
        scores, maps = runtime.score_and_maps(fitted_case.inputs, include_maps=True)
    finally:
        runtime.close()
    np.testing.assert_allclose(
        scores,
        fitted_case.reference["score"],
        atol=1e-5,
        rtol=1e-4,
    )
    assert maps is not None
    np.testing.assert_allclose(
        maps,
        fitted_case.reference["anomaly_map"],
        atol=1e-5,
        rtol=1e-4,
    )


def test_native_export_uses_registered_codec_and_public_loader(
    fitted_case: _FittedCase,
    tmp_path: Path,
) -> None:
    from pyimgano.inference import load_artifact

    out = tmp_path / "native"
    fitted_case.adapter.export_artifact(
        fitted_case.detector,
        format=ArtifactFormat.NATIVE,
        context=replace(
            fitted_case.context,
            output_contract=fitted_case.adapter.build_output_contract(),
            compatibility={"platforms": ["windows-x86_64"]},
        ),
        out=out,
    )
    from pyimgano.artifacts import load_artifact_manifest
    from pyimgano.artifacts.compatibility import current_platform_tag

    manifest = load_artifact_manifest(out)
    assert manifest["compatibility"]["platforms"] == [current_platform_tag()]
    assert manifest["compatibility"]["runtime_versions"] == {"torch": ">=1.9"}
    runtime = load_artifact(out, format="native", backend="pyimgano")
    try:
        scores, maps = runtime.score_and_maps(fitted_case.inputs, include_maps=True)
    finally:
        runtime.close()
    np.testing.assert_allclose(
        scores,
        fitted_case.reference["score"],
        atol=1e-5,
        rtol=1e-4,
    )
    assert maps is not None
    np.testing.assert_allclose(
        maps,
        fitted_case.reference["anomaly_map"],
        atol=1e-5,
        rtol=1e-4,
    )


def test_openvino_export_is_conditional_and_verified_from_onnx(
    fitted_case: _FittedCase,
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("openvino")
    from pyimgano.artifacts import load_artifact_manifest
    from pyimgano.inference import load_artifact

    out = tmp_path / "openvino"
    fitted_case.adapter.export_artifact(
        fitted_case.detector,
        format=ArtifactFormat.OPENVINO,
        context=fitted_case.context,
        out=out,
    )
    manifest = load_artifact_manifest(out)
    by_role = {component["role"]: component for component in manifest["components"]}
    assert by_role["runtime_model"]["path"] == "model/detector.xml"
    assert by_role["openvino_weights"]["path"] == "model/detector.bin"
    report_path = out / manifest["verification"]["report"]["path"]
    assert "verified_onnx_source_parity" in report_path.read_text(encoding="utf-8")

    runtime = load_artifact(out, format="openvino-ir", backend="openvino", device="CPU")
    try:
        scores, maps = runtime.score_and_maps(fitted_case.inputs, include_maps=True)
    finally:
        runtime.close()
    np.testing.assert_allclose(
        scores,
        fitted_case.reference["score"],
        atol=1e-5,
        rtol=1e-4,
    )
    assert maps is not None
    np.testing.assert_allclose(
        maps,
        fitted_case.reference["anomaly_map"],
        atol=1e-5,
        rtol=1e-4,
    )
