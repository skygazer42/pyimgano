from __future__ import annotations

import numpy as np

from pyimgano.exporting.state_codec import inspect_checkpoint_contract
from pyimgano.exporting.types import CheckpointCompleteness, CheckpointContract
from pyimgano.serialization.safe_detector_state import (
    inspect_safe_detector_state,
    save_safe_detector_state,
)
from pyimgano.training.checkpointing import (
    build_checkpoint_contract,
    failed_checkpoint_contract,
)
from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector


class _Detector:
    def __init__(self, value: int) -> None:
        self.value = np.asarray([value], dtype=np.int64)


def test_legacy_safe_checkpoint_stays_unknown_after_successful_restore(tmp_path) -> None:
    path = save_safe_detector_state(_Detector(7), tmp_path / "legacy.pyim")
    before = inspect_checkpoint_contract(path)
    target = _Detector(0)

    load_checkpoint_into_detector(target, path)
    after = inspect_checkpoint_contract(path)

    assert target.value.tolist() == [7]
    assert before.completeness is CheckpointCompleteness.UNKNOWN
    assert after.completeness is CheckpointCompleteness.UNKNOWN
    assert before.roundtrip_verified is False
    assert after.roundtrip_verified is False
    assert inspect_safe_detector_state(path)["completeness"] == "unknown"


def test_missing_checkpoint_contract_fields_default_to_unknown() -> None:
    contract = CheckpointContract.from_mapping({"path": "checkpoints/model.pt"})

    assert contract.completeness is CheckpointCompleteness.UNKNOWN
    assert contract.strict_exportable is False


def test_loadable_structured_checkpoint_does_not_self_attest_completeness(tmp_path) -> None:
    from pyimgano.serialization.safe_checkpoint import save_safe_checkpoint

    path = save_safe_checkpoint(
        {"format": "model-specific", "weights": np.arange(3, dtype=np.float32)},
        tmp_path / "model.pyim",
    )

    contract = inspect_checkpoint_contract(path)

    assert contract.completeness is CheckpointCompleteness.UNKNOWN
    assert contract.sha256 is not None
    assert contract.size_bytes == path.stat().st_size


def test_explicit_verified_checkpoint_contract_is_exportable_and_canonical(tmp_path) -> None:
    checkpoint = tmp_path / "model.pyim"
    checkpoint.write_bytes(b"safe checkpoint payload")
    kwargs = {
        "path": checkpoint,
        "codec_id": "test-codec",
        "codec_version": 1,
        "adapter_id": "test-adapter",
        "adapter_version": 2,
        "state_schema_version": 3,
        "roundtrip_verified": True,
        "roundtrip": {"probe": "passed"},
    }

    first = build_checkpoint_contract(
        model_config={"model": "demo", "kwargs": {"alpha": 1, "beta": 2}},
        **kwargs,
    )
    reordered = build_checkpoint_contract(
        model_config={"kwargs": {"beta": 2, "alpha": 1}, "model": "demo"},
        **kwargs,
    )

    assert first.completeness is CheckpointCompleteness.COMPLETE
    assert first.strict_exportable is True
    assert first.model_config_fingerprint == reordered.model_config_fingerprint
    assert first.sha256 == reordered.sha256
    assert first.size_bytes == checkpoint.stat().st_size


def test_unverified_checkpoint_contract_is_partial_not_exportable(tmp_path) -> None:
    checkpoint = tmp_path / "model.pyim"
    checkpoint.write_bytes(b"safe checkpoint payload")

    contract = build_checkpoint_contract(
        checkpoint,
        codec_id="test-codec",
        codec_version=1,
        adapter_id="test-adapter",
        adapter_version=1,
        model_config={"model": "demo"},
        state_schema_version=1,
        roundtrip_verified=False,
        roundtrip={"probe": "not-run"},
    )

    assert contract.completeness is CheckpointCompleteness.PARTIAL
    assert contract.strict_exportable is False


def test_failed_checkpoint_contract_never_becomes_exportable() -> None:
    contract = failed_checkpoint_contract("restore probe failed")

    assert contract.completeness is CheckpointCompleteness.FAILED
    assert contract.roundtrip_verified is False
    assert contract.strict_exportable is False
    assert contract.failure_reason == "restore probe failed"
