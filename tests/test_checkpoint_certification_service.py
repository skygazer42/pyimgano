from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _Detector:
    def __init__(self, weights) -> None:  # noqa: ANN001
        self.weights = np.asarray(weights, dtype=np.float32)
        self.threshold_ = 100.0

    def decision_function(self, inputs):  # noqa: ANN001
        return np.asarray(
            [
                float(np.asarray(item, dtype=np.float32).mean() * self.weights.mean())
                for item in inputs
            ],
            dtype=np.float32,
        )


class _Adapter:
    adapter_id = "test.certified-adapter"
    adapter_version = 1
    model_names = ("test_certified_model",)
    state_codec_id = "test.certified-state"

    def build_probe_spec(self, detector, *, context=None):  # noqa: ANN001
        from pyimgano.exporting import ProbeSpec

        _ = detector, context
        return ProbeSpec(
            inputs=(
                np.zeros((3, 4, 3), dtype=np.uint8),
                np.full((3, 4, 3), 5, dtype=np.uint8),
            )
        )

    def verify_roundtrip(self, original, restored, spec):  # noqa: ANN001
        before = original.decision_function(list(spec.inputs))
        after = restored.decision_function(list(spec.inputs))
        max_error = float(np.max(np.abs(before - after)))
        return {"passed": bool(max_error <= 1e-6), "max_score_abs_error": max_error}


@pytest.fixture
def certification_registries(monkeypatch):  # noqa: ANN001
    import pyimgano.exporting.registry as adapter_module
    import pyimgano.exporting.state_codec as codec_module
    import pyimgano.services.checkpoint_certification_service as service
    from pyimgano.exporting import MappingStateCodec, StateField
    from pyimgano.exporting.registry import ExportAdapterRegistry
    from pyimgano.exporting.state_codec import StateCodecRegistry

    adapters = ExportAdapterRegistry()
    adapters.register(_Adapter())
    codecs = StateCodecRegistry()
    codecs.register(
        MappingStateCodec(
            codec_id="test.certified-state",
            codec_version=1,
            state_schema_version=1,
            model_names=("test_certified_model",),
            fields=(StateField("weights", dtypes=("float32",), ranks=(1,)),),
        )
    )
    monkeypatch.setattr(adapter_module, "EXPORT_ADAPTER_REGISTRY", adapters)
    monkeypatch.setattr(codec_module, "STATE_CODEC_REGISTRY", codecs)
    monkeypatch.setattr(service, "_load_builtin_adapters", lambda: None)
    monkeypatch.setattr(service, "_fresh_detector", lambda config: _Detector([0.0, 0.0]))
    monkeypatch.setattr(service, "_resolved_model_kwargs", lambda config: {"device": "cpu"})
    return adapters, codecs


def _config():
    return SimpleNamespace(
        seed=1,
        model=SimpleNamespace(
            name="test_certified_model",
            model_kwargs={},
            device="cpu",
            contamination=0.1,
            pretrained=False,
            preset=None,
        ),
    )


def test_certification_replaces_source_with_complete_safe_fitted_state(
    tmp_path, certification_registries
) -> None:
    from pyimgano.exporting import inspect_fitted_state, load_fitted_state
    from pyimgano.services.checkpoint_certification_service import certify_checkpoint_for_export

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"opaque-source-checkpoint")
    detector = _Detector([2.0, 4.0])

    contract = certify_checkpoint_for_export(detector, checkpoint, config=_config())

    assert contract is not None and contract.strict_exportable is True
    info = inspect_fitted_state(checkpoint)
    assert info.codec_id == "test.certified-state"
    restored = _Detector([0.0, 0.0])
    load_fitted_state(restored, checkpoint, expected_model_name="test_certified_model")
    np.testing.assert_allclose(restored.weights, detector.weights)
    assert restored.threshold_ == 100.0


def test_certification_failure_does_not_replace_source(
    tmp_path, certification_registries, monkeypatch
) -> None:
    import pyimgano.exporting.registry as adapter_module
    from pyimgano.services.checkpoint_certification_service import (
        CheckpointCertificationError,
        certify_checkpoint_for_export,
    )

    adapter = adapter_module.EXPORT_ADAPTER_REGISTRY.get("test_certified_model")
    monkeypatch.setattr(
        adapter,
        "verify_roundtrip",
        lambda original, restored, spec: {"passed": False, "reason": "mismatch"},
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"keep-original")

    with pytest.raises(CheckpointCertificationError, match="parity probe failed"):
        certify_checkpoint_for_export(_Detector([1.0, 2.0]), checkpoint, config=_config())

    assert checkpoint.read_bytes() == b"keep-original"
