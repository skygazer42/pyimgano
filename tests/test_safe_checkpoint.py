from __future__ import annotations

import zipfile

import numpy as np
import pytest

from pyimgano.serialization.safe_checkpoint import (
    SafeCheckpointError,
    load_safe_checkpoint,
    save_safe_checkpoint,
)


class _LegacyDetector:
    def __init__(self, marker: str) -> None:
        self.marker = marker


def test_safe_checkpoint_roundtrip_preserves_structured_values(tmp_path) -> None:
    path = tmp_path / "model.ckpt"
    payload = {
        "schema_version": 1,
        "config": {"name": "demo", "shape": (2, 3), "enabled": True},
        "state": {
            "weights": np.arange(6, dtype=np.float32).reshape(2, 3),
            "labels": np.asarray([0, 1], dtype=np.int64),
            "optional": None,
        },
    }

    save_safe_checkpoint(payload, path)
    loaded = load_safe_checkpoint(path)

    assert loaded["schema_version"] == 1
    assert loaded["config"]["shape"] == (2, 3)
    np.testing.assert_array_equal(loaded["state"]["weights"], payload["state"]["weights"])
    np.testing.assert_array_equal(loaded["state"]["labels"], payload["state"]["labels"])


def test_safe_checkpoint_rejects_arbitrary_python_objects(tmp_path) -> None:
    with pytest.raises(SafeCheckpointError, match="Unsupported value"):
        save_safe_checkpoint({"unsafe": object()}, tmp_path / "unsafe.ckpt")


def test_safe_checkpoint_detects_array_tampering(tmp_path) -> None:
    path = tmp_path / "model.ckpt"
    save_safe_checkpoint({"weights": np.arange(4, dtype=np.float32)}, path)

    tampered_path = tmp_path / "tampered.ckpt"
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(tampered_path, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "arrays/00000000.npy":
                data = b"tampered"
            target.writestr(info, data)

    with pytest.raises(SafeCheckpointError, match="size mismatch"):
        load_safe_checkpoint(tampered_path)


def test_safe_checkpoint_rejects_pickle_payload_without_executing_it(tmp_path) -> None:
    marker = tmp_path / "executed"
    path = tmp_path / "malicious.ckpt"
    path.write_bytes(b"not-a-zip-pickle-payload:" + str(marker).encode("utf-8"))

    with pytest.raises(SafeCheckpointError, match="not a valid safe archive"):
        load_safe_checkpoint(path)
    assert not marker.exists()


def test_joblib_model_loader_requires_trust_and_checks_digest(tmp_path) -> None:
    from pyimgano.models.serialization import UntrustedModelArtifactError, load_model, save_model

    path = save_model({"value": 7}, tmp_path / "legacy.joblib")
    assert path.with_name(f"{path.name}.sha256").is_file()

    with pytest.raises(UntrustedModelArtifactError, match="trusted=True"):
        load_model(path)
    assert load_model(path, trusted=True) == {"value": 7}

    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_model(path, trusted=True)


def test_workbench_legacy_joblib_restore_is_fail_closed(tmp_path) -> None:
    from pyimgano.models.serialization import save_model
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    path = save_model(_LegacyDetector("trained"), tmp_path / "legacy.joblib")
    target = _LegacyDetector("fresh")

    with pytest.raises(NotImplementedError, match="requires trusted=True"):
        load_checkpoint_into_detector(target, path)
    assert target.marker == "fresh"

    load_checkpoint_into_detector(target, path, trusted=True)
    assert target.marker == "trained"


def test_workbench_safe_detector_state_restore_needs_no_trust(tmp_path) -> None:
    from pyimgano.serialization.safe_detector_state import save_safe_detector_state
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    path = save_safe_detector_state(_LegacyDetector("trained"), tmp_path / "safe.ckpt")
    target = _LegacyDetector("fresh")

    load_checkpoint_into_detector(target, path)
    assert target.marker == "trained"
