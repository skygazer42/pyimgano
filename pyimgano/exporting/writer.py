from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping


class ArtifactWriteError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactWriter:
    """Build and atomically publish one artifact directory.

    Manifest construction and validation stay owned by :mod:`pyimgano.artifacts`;
    this class only stages bytes and controls the filesystem transaction.
    """

    def __init__(self, out: str | Path, *, overwrite: bool = False) -> None:
        self.out = Path(out)
        self.overwrite = bool(overwrite)
        self._staging_root: Path | None = None
        self._committed = False

    @property
    def root(self) -> Path:
        if self._staging_root is None:
            raise ArtifactWriteError("ArtifactWriter has not been entered.")
        return self._staging_root

    def __enter__(self) -> "ArtifactWriter":
        if self._staging_root is not None:
            raise ArtifactWriteError("ArtifactWriter cannot be entered twice.")
        self.out.parent.mkdir(parents=True, exist_ok=True)
        if self.out.exists() and not self.overwrite:
            raise FileExistsError(f"Artifact output already exists: {self.out}")
        staging = tempfile.mkdtemp(
            prefix=f".{self.out.name}.staging-",
            dir=str(self.out.parent),
        )
        self._staging_root = Path(staging)
        return self

    def _path_for(self, relative_path: str) -> Path:
        from pyimgano.artifacts import resolve_contained_path

        return resolve_contained_path(self.root, str(relative_path), must_exist=False)

    def path_for(self, relative_path: str) -> Path:
        return self._path_for(relative_path)

    def write_bytes(self, relative_path: str, data: bytes) -> Path:
        target = self._path_for(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ArtifactWriteError(f"Artifact path already written: {relative_path}")
        target.write_bytes(bytes(data))
        return target

    def write_json(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        from pyimgano.artifacts import canonical_json_bytes

        return self.write_bytes(relative_path, canonical_json_bytes(dict(payload)))

    def copy_file(self, source: str | Path, relative_path: str) -> Path:
        source_path = Path(source)
        target = self._path_for(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ArtifactWriteError(f"Artifact path already written: {relative_path}")
        from pyimgano.artifacts.security import (
            ArtifactSecurityError,
            copy_regular_file_nofollow,
        )

        try:
            copy_regular_file_nofollow(source_path, target)
        except (ArtifactSecurityError, OSError) as exc:
            raise ArtifactWriteError(
                f"Artifact component source must be a stable regular file: {source_path}: {exc}"
            ) from exc
        return target

    def component_metadata(
        self,
        relative_path: str,
        *,
        component_id: str,
        role: str,
        format: str,
        serialization: str,
    ) -> dict[str, Any]:
        path = self._path_for(relative_path)
        if path.is_symlink() or not path.is_file():
            raise ArtifactWriteError(f"Artifact component was not written: {relative_path}")
        return {
            "id": str(component_id),
            "path": str(relative_path),
            "role": str(role),
            "format": str(format),
            "serialization": str(serialization),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    def finalize(
        self,
        manifest_payload: Mapping[str, Any],
        *,
        policy: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        if self._committed:
            raise ArtifactWriteError("ArtifactWriter has already committed its output.")
        from pyimgano.artifacts import (
            load_artifact_manifest,
            verify_artifact_files,
            write_artifact_manifest,
        )

        write_artifact_manifest(self.root, dict(manifest_payload), policy=dict(policy))
        manifest = load_artifact_manifest(self.root)
        verify_artifact_files(self.root, manifest)

        backup: Path | None = None
        try:
            if self.out.exists():
                if not self.overwrite:
                    raise FileExistsError(f"Artifact output already exists: {self.out}")
                backup = self.out.with_name(f".{self.out.name}.backup-{uuid.uuid4().hex}")
                os.replace(self.out, backup)
            os.replace(self.root, self.out)
            self._committed = True
            self._staging_root = None
        except Exception:
            if backup is not None and backup.exists() and not self.out.exists():
                os.replace(backup, self.out)
            raise
        else:
            if backup is not None and backup.exists():
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
        return self.out / "artifact_manifest.json", dict(manifest)

    def abort(self) -> None:
        if self._staging_root is not None and self._staging_root.exists():
            shutil.rmtree(self._staging_root)
        self._staging_root = None

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        if not self._committed:
            self.abort()


__all__ = ["ArtifactWriteError", "ArtifactWriter", "sha256_file"]
