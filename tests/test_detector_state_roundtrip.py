from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from pyimgano.exporting.state_codec import (
    MappingStateCodec,
    StateCodecError,
    StateCodecRegistry,
    StateField,
    inspect_fitted_state,
    load_fitted_state,
    save_fitted_state,
)
from pyimgano.exporting.types import (
    CheckpointCompleteness,
    CheckpointContract,
)


class _Detector:
    def __init__(self, weights, *, threshold: float) -> None:  # noqa: ANN001
        self.weights = np.asarray(weights, dtype=np.float32)
        self.threshold_ = float(threshold)

    def decision_function(self, values):  # noqa: ANN001
        return np.asarray(values, dtype=np.float32) @ self.weights


def _contract() -> CheckpointContract:
    return CheckpointContract(
        completeness=CheckpointCompleteness.COMPLETE,
        codec_id="test.weights",
        codec_version=1,
        adapter_id="test.reference",
        adapter_version=1,
        model_config_fingerprint="sha256:" + "1" * 64,
        state_schema_version=1,
        size_bytes=4,
        sha256=hashlib.sha256(b"source").hexdigest(),
        roundtrip_verified=True,
        roundtrip={"probe": "passed"},
    )


@pytest.fixture
def isolated_codec_registry(monkeypatch):  # noqa: ANN001
    import pyimgano.exporting.state_codec as module

    registry = StateCodecRegistry()
    monkeypatch.setattr(module, "STATE_CODEC_REGISTRY", registry)
    registry.register(
        MappingStateCodec(
            codec_id="test.weights",
            codec_version=1,
            state_schema_version=1,
            model_names=("test_model",),
            fields=(
                StateField(
                    "weights",
                    dtypes=("float32",),
                    ranks=(1,),
                    max_bytes=1024,
                ),
            ),
        )
    )
    return registry


def test_registered_state_codec_roundtrip_excludes_operating_threshold(
    tmp_path,
    isolated_codec_registry,
) -> None:
    source = _Detector([1.0, 2.0], threshold=0.75)
    path = save_fitted_state(
        source,
        tmp_path / "state.pyim",
        model_name="test_model",
        checkpoint_contract=_contract(),
    )
    restored = _Detector([0.0, 0.0], threshold=-123.0)

    load_fitted_state(restored, path, expected_model_name="test_model")

    np.testing.assert_allclose(restored.weights, source.weights)
    assert restored.threshold_ == -123.0
    assert inspect_fitted_state(path).completeness is CheckpointCompleteness.COMPLETE


def test_unknown_checkpoint_cannot_be_saved_as_complete_fitted_state(
    tmp_path,
    isolated_codec_registry,
) -> None:
    source = _Detector([1.0, 2.0], threshold=0.5)
    unknown = replace(
        _contract(),
        completeness=CheckpointCompleteness.UNKNOWN,
        roundtrip_verified=False,
    )

    with pytest.raises(StateCodecError, match="complete, verified"):
        save_fitted_state(
            source,
            tmp_path / "state.pyim",
            model_name="test_model",
            checkpoint_contract=unknown,
        )
    assert not (tmp_path / "state.pyim").exists()


def test_mapping_codec_rejects_dtype_rank_extra_fields_and_threshold() -> None:
    codec = MappingStateCodec(
        codec_id="test.weights",
        codec_version=1,
        state_schema_version=1,
        model_names=("test_model",),
        fields=(StateField("weights", dtypes=("float32",), ranks=(1,)),),
    )

    with pytest.raises(StateCodecError, match="dtype"):
        codec.validate_state({"weights": np.asarray([1], dtype=np.int64)})
    with pytest.raises(StateCodecError, match="rank"):
        codec.validate_state({"weights": np.ones((1, 1), dtype=np.float32)})
    with pytest.raises(StateCodecError, match="unregistered"):
        codec.validate_state({"weights": np.ones(1, dtype=np.float32), "other": np.ones(1)})
    with pytest.raises(ValueError, match="Operating-policy"):
        StateField("threshold_")
