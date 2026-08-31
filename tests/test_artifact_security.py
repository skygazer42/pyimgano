from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "a/../b", "/tmp/model", "a\\b", "a//b", "C:/model.onnx"],
)
def test_resolve_contained_path_rejects_non_portable_or_escaping_paths(
    tmp_path: Path, value: str
) -> None:
    from pyimgano.artifacts.security import ArtifactSecurityError, resolve_contained_path

    with pytest.raises(ArtifactSecurityError):
        resolve_contained_path(tmp_path, value, must_exist=False)


def test_resolve_contained_path_rejects_symlink_components(tmp_path: Path) -> None:
    from pyimgano.artifacts.security import ArtifactSecurityError, resolve_contained_path

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.onnx").write_bytes(b"model")
    root = tmp_path / "artifact"
    root.mkdir()
    try:
        (root / "model").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ArtifactSecurityError, match="symlink"):
        resolve_contained_path(root, "model/model.onnx")


def test_verify_file_enforces_size_and_hash(tmp_path: Path) -> None:
    from pyimgano.artifacts.security import ArtifactSecurityError, verify_file

    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    assert verify_file(path, size_bytes=5, sha256=_sha(b"model")) == path

    with pytest.raises(ArtifactSecurityError, match="size"):
        verify_file(path, size_bytes=4, sha256=_sha(b"model"))
    with pytest.raises(ArtifactSecurityError, match="SHA-256"):
        verify_file(path, size_bytes=5, sha256="0" * 64)


def test_verify_artifact_files_checks_components_policy_and_attachments(tmp_path: Path) -> None:
    from pyimgano.artifacts.security import ArtifactSecurityError, verify_artifact_files

    root = tmp_path / "artifact"
    (root / "model").mkdir(parents=True)
    (root / "verification").mkdir()
    (root / "model" / "detector.onnx").write_bytes(b"model")
    (root / "infer_config.json").write_bytes(b"{}")
    (root / "verification" / "parity.json").write_bytes(b"report")
    manifest = {
        "components": [
            {
                "path": "model/detector.onnx",
                "size_bytes": 5,
                "sha256": _sha(b"model"),
            }
        ],
        "policy_ref": {"path": "infer_config.json", "sha256": _sha(b"{}")},
        "verification": {
            "report": {
                "path": "verification/parity.json",
                "size_bytes": 6,
                "sha256": _sha(b"report"),
            }
        },
    }
    verified = verify_artifact_files(root, manifest)
    assert set(verified) == {
        "model/detector.onnx",
        "infer_config.json",
        "verification/parity.json",
    }

    (root / "verification" / "parity.json").write_bytes(b"tamper")
    with pytest.raises(ArtifactSecurityError, match="SHA-256"):
        verify_artifact_files(root, manifest)


def test_stage_verified_artifact_uses_private_verified_bytes(tmp_path: Path) -> None:
    from pyimgano.artifacts.security import stage_verified_artifact

    root = tmp_path / "artifact"
    (root / "model").mkdir(parents=True)
    source = root / "model" / "detector.onnx"
    source.write_bytes(b"verified")
    manifest = {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "components": [
            {
                "path": "model/detector.onnx",
                "size_bytes": 8,
                "sha256": _sha(b"verified"),
            }
        ],
    }

    with stage_verified_artifact(root, manifest) as staging:
        staged_path = staging.path_for("model/detector.onnx")
        assert staged_path.read_bytes() == b"verified"
        source.write_bytes(b"replaced")
        assert staged_path.read_bytes() == b"verified"
        assert not (stat.S_IMODE(staged_path.stat().st_mode) & stat.S_IWUSR)
        staged_root = staging.root
    assert not staged_root.exists()


def test_stage_verified_artifact_rejects_source_symlink(tmp_path: Path) -> None:
    from pyimgano.artifacts.security import ArtifactSecurityError, stage_verified_artifact

    root = tmp_path / "artifact"
    root.mkdir()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"verified")
    try:
        (root / "model.onnx").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    manifest = {
        "components": [{"path": "model.onnx", "size_bytes": 8, "sha256": _sha(b"verified")}]
    }
    with pytest.raises(ArtifactSecurityError, match="symlink"):
        stage_verified_artifact(root, manifest)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission assertion")
def test_staging_root_is_not_user_writable_while_active(tmp_path: Path) -> None:
    from pyimgano.artifacts.security import stage_verified_artifact

    root = tmp_path / "artifact"
    root.mkdir()
    (root / "state.bin").write_bytes(b"state")
    manifest = {"components": [{"path": "state.bin", "size_bytes": 5, "sha256": _sha(b"state")}]}
    with stage_verified_artifact(root, manifest) as staging:
        assert not (stat.S_IMODE(staging.root.stat().st_mode) & stat.S_IWUSR)


def test_artifact_writer_copy_file_rejects_source_symlink(tmp_path: Path) -> None:
    from pyimgano.exporting.writer import ArtifactWriteError, ArtifactWriter

    source = tmp_path / "source.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with ArtifactWriter(tmp_path / "artifact") as writer:
        with pytest.raises(ArtifactWriteError, match="symlink"):
            writer.copy_file(source, "model/source.bin")


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX rename of an open file")
def test_artifact_writer_copy_file_consumes_one_open_descriptor(
    monkeypatch, tmp_path: Path
) -> None:
    from pyimgano.artifacts import security
    from pyimgano.exporting.writer import ArtifactWriter

    source = tmp_path / "source.bin"
    saved = tmp_path / "saved.bin"
    source.write_bytes(b"verified source bytes")
    real_copy = security._copy_open_source_file
    swapped = False

    def replace_after_open(descriptor, destination, *, maximum_bytes, label):
        nonlocal swapped
        if not swapped:
            swapped = True
            source.rename(saved)
            source.write_bytes(b"replacement bytes")
        return real_copy(
            descriptor,
            destination,
            maximum_bytes=maximum_bytes,
            label=label,
        )

    monkeypatch.setattr(security, "_copy_open_source_file", replace_after_open)
    with ArtifactWriter(tmp_path / "artifact") as writer:
        copied = writer.copy_file(source, "model/source.bin")
        assert copied.read_bytes() == b"verified source bytes"
        assert swapped
