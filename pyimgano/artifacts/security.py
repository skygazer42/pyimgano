from __future__ import annotations

"""Filesystem containment and verified-byte staging for artifact runtimes."""

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Mapping

from pyimgano.artifacts.onnx_external_data import external_data_locations

_COPY_CHUNK_BYTES = 1024 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_ONNX_PROTOBUF_BYTES = 512 * 1024 * 1024


class ArtifactSecurityError(ValueError):
    """Raised when artifact files violate containment or integrity rules."""


def _relative_path(value: str | Path) -> str:
    if not isinstance(value, (str, Path)):
        raise ArtifactSecurityError("Artifact path must be a string or Path.")
    text = str(value)
    if not text or text != text.strip():
        raise ArtifactSecurityError("Artifact path must be non-empty without surrounding space.")
    if "\x00" in text or "\\" in text or "//" in text or text.endswith("/"):
        raise ArtifactSecurityError(
            f"Artifact path must be a normalized relative POSIX path: {text!r}"
        )
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ArtifactSecurityError(f"Absolute or drive-qualified artifact path: {text!r}")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise ArtifactSecurityError(f"Artifact path contains dot/dot-dot/empty segment: {text!r}")
    return text


def _ensure_root(root: str | Path) -> Path:
    value = Path(root)
    if value.is_symlink():
        raise ArtifactSecurityError(f"Artifact root must not be a symlink: {value}")
    if not value.is_dir():
        raise ArtifactSecurityError(f"Artifact root is not a directory: {value}")
    return value.resolve()


def resolve_contained_path(
    root: str | Path,
    relative_path: str | Path,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one strict relative POSIX path without following symlink components."""

    root_path = _ensure_root(root)
    relative = _relative_path(relative_path)
    candidate = root_path.joinpath(*relative.split("/"))

    current = root_path
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if must_exist or index < len(parts) - 1:
                # Missing ancestors are allowed only for a writer after it creates
                # them explicitly.  Existing path validation stays fail-closed.
                if must_exist:
                    raise ArtifactSecurityError(f"Artifact file is missing: {relative}") from None
            break
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactSecurityError(f"Artifact path contains a symlink: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ArtifactSecurityError(
                f"Artifact path ancestor is not a directory: {'/'.join(parts[: index + 1])}"
            )

    resolved_parent = candidate.parent.resolve(strict=False)
    try:
        if os.path.commonpath([str(root_path), str(resolved_parent)]) != str(root_path):
            raise ArtifactSecurityError(f"Artifact path escapes its root: {relative}")
    except ValueError as exc:
        raise ArtifactSecurityError(f"Artifact path escapes its root: {relative}") from exc

    if must_exist:
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            raise ArtifactSecurityError(f"Artifact file is missing: {relative}") from None
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactSecurityError(f"Artifact file is a symlink: {relative}")
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactSecurityError(f"Artifact path is not a regular file: {relative}")
    return candidate


def _validate_expected(*, size_bytes: int | None, sha256: str) -> tuple[int | None, str]:
    if size_bytes is not None and (
        not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0
    ):
        raise ArtifactSecurityError("Expected file size must be a non-negative integer.")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in _HEX_DIGITS for char in sha256)
    ):
        raise ArtifactSecurityError("Expected SHA-256 must be 64 lowercase hexadecimal characters.")
    return size_bytes, sha256


def _open_nofollow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactSecurityError(f"Cannot securely open artifact file {path}: {exc}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ArtifactSecurityError(f"Artifact path is not a regular file: {path}")
    return descriptor


def _open_contained_nofollow(root: Path, relative: str) -> int:
    """Open a contained file without a path-resolution race on POSIX."""

    if os.name != "posix" or os.open not in getattr(os, "supports_dir_fd", set()):
        return _open_nofollow(resolve_contained_path(root, relative))

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        file_flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW

    parts = _relative_path(relative).split("/")
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise ArtifactSecurityError(f"Cannot securely open artifact root {root}: {exc}") from exc
    try:
        for part in parts[:-1]:
            try:
                child_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ArtifactSecurityError(
                    f"Cannot securely traverse artifact path {relative!r}: {exc}"
                ) from exc
            os.close(directory_fd)
            directory_fd = child_fd
        try:
            descriptor = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise ArtifactSecurityError(
                f"Cannot securely open artifact file {relative!r}: {exc}"
            ) from exc
    finally:
        os.close(directory_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ArtifactSecurityError(f"Artifact path is not a regular file: {relative}")
    return descriptor


def _secure_source_openat_available() -> bool:
    """Return whether this platform can traverse a source tree race-free.

    Source ingestion must not silently fall back to a check-then-open sequence.
    In particular, Python's Windows ``os.open`` API has no dir-fd traversal or
    no-follow flag that is equivalent to POSIX ``openat(..., O_NOFOLLOW)``.
    Callers therefore fail closed when these primitives are unavailable.
    """

    return bool(
        os.name == "posix"
        and os.open in getattr(os, "supports_dir_fd", set())
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


def _source_directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _source_file_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _open_source_directory(root: str | Path) -> tuple[Path, int]:
    if not _secure_source_openat_available():
        raise ArtifactSecurityError(
            "Secure source ingestion requires POSIX openat/O_NOFOLLOW support; "
            "this platform fails closed instead of using a racy resolve/open fallback."
        )
    try:
        resolved = Path(root).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ArtifactSecurityError(f"Secure source root is unavailable: {root}") from exc
    if not resolved.is_absolute():  # pragma: no cover - Path.resolve() is absolute
        raise ArtifactSecurityError(f"Secure source root must be absolute: {resolved}")

    flags = _source_directory_flags()
    anchor = resolved.anchor
    try:
        directory_fd = os.open(anchor, flags)
    except OSError as exc:
        raise ArtifactSecurityError(
            f"Cannot securely open source filesystem anchor {anchor!r}: {exc}"
        ) from exc
    try:
        for part in resolved.parts[1:]:
            try:
                child_fd = os.open(part, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ArtifactSecurityError(
                    f"Cannot securely traverse source root {resolved}: {exc}"
                ) from exc
            os.close(directory_fd)
            directory_fd = child_fd
        info = os.fstat(directory_fd)
        if not stat.S_ISDIR(info.st_mode):
            raise ArtifactSecurityError(f"Secure source root is not a directory: {resolved}")
        return resolved, directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _source_relative_parts(relative_path: str | Path) -> tuple[str, ...]:
    text = str(relative_path)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or "\x00" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
    ):
        raise ArtifactSecurityError(f"Unsafe relative source path: {text!r}")
    parts = tuple(text.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactSecurityError(f"Unsafe relative source path: {text!r}")
    return parts


def _open_source_regular_at(directory_fd: int, name: str, *, display: str) -> int:
    """Open one leaf relative to an already-stable directory descriptor."""

    try:
        descriptor = os.open(name, _source_file_flags(), dir_fd=directory_fd)
    except FileNotFoundError as exc:
        raise ArtifactSecurityError(f"Source file is missing: {display}") from exc
    except OSError as exc:
        raise ArtifactSecurityError(
            f"Cannot securely open source file {display!r}; symlinks are forbidden: {exc}"
        ) from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ArtifactSecurityError(f"Source path is not a regular file: {display}")
    return descriptor


def _open_source_file(root_fd: int, relative_path: str | Path) -> int:
    parts = _source_relative_parts(relative_path)
    display = str(relative_path)
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            try:
                child_fd = os.open(part, _source_directory_flags(), dir_fd=directory_fd)
            except OSError as exc:
                raise ArtifactSecurityError(
                    f"Cannot securely traverse source path {display!r}; "
                    f"symlinks are forbidden: {exc}"
                ) from exc
            os.close(directory_fd)
            directory_fd = child_fd
        return _open_source_regular_at(directory_fd, parts[-1], display=display)
    finally:
        os.close(directory_fd)


def _source_snapshot(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        int(getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000))),
    )


def _copy_open_source_file(
    descriptor: int,
    destination: Path,
    *,
    maximum_bytes: int | None,
    label: str,
) -> tuple[int, str]:
    """Consume one open source descriptor and copy exactly those bytes once."""

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactSecurityError(f"Source path is not a regular file: {label}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ArtifactSecurityError(
                f"Source file exceeds the safe size limit of {maximum_bytes} bytes: {label}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        os.close(descriptor)
        raise

    digest = hashlib.sha256()
    total = 0
    created = False
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source_handle:
            with destination.open("xb") as destination_handle:
                created = True
                while True:
                    chunk = source_handle.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if maximum_bytes is not None and total > maximum_bytes:
                        raise ArtifactSecurityError(
                            f"Source file exceeds the safe size limit of "
                            f"{maximum_bytes} bytes: {label}"
                        )
                    destination_handle.write(chunk)
                    digest.update(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
                after = os.fstat(source_handle.fileno())
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise

    if _source_snapshot(before) != _source_snapshot(after):
        destination.unlink(missing_ok=True)
        raise ArtifactSecurityError(f"Source file changed while it was being copied: {label}")
    return total, digest.hexdigest()


class SecureSourceTree:
    """Stable no-follow view used to ingest files from one source directory.

    The root directory descriptor remains open across all copies, so replacing a
    lexical source directory cannot redirect later dependency reads elsewhere.
    """

    def __init__(self, root: str | Path) -> None:
        self.root, self._descriptor = _open_source_directory(root)
        self._closed = False

    def copy_file(
        self,
        relative_path: str | Path,
        destination: str | Path,
        *,
        maximum_bytes: int | None = None,
    ) -> tuple[int, str]:
        if self._closed:
            raise ArtifactSecurityError("Secure source tree has already been closed.")
        if maximum_bytes is not None and (
            not isinstance(maximum_bytes, int)
            or isinstance(maximum_bytes, bool)
            or maximum_bytes < 0
        ):
            raise ArtifactSecurityError("maximum_bytes must be a non-negative integer.")
        relative = str(relative_path)
        descriptor = _open_source_file(self._descriptor, relative)
        return _copy_open_source_file(
            descriptor,
            Path(destination),
            maximum_bytes=maximum_bytes,
            label=relative,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._descriptor)

    def __enter__(self) -> "SecureSourceTree":
        if self._closed:
            raise ArtifactSecurityError("Secure source tree has already been closed.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def copy_regular_file_nofollow(
    source: str | Path,
    destination: str | Path,
    *,
    maximum_bytes: int | None = None,
) -> tuple[int, str]:
    """Copy a regular file through one no-follow descriptor into a new file."""

    source_path = Path(source)
    if source_path.is_symlink():
        raise ArtifactSecurityError(f"Source file must not be a symlink: {source_path}")
    with SecureSourceTree(source_path.parent) as source_tree:
        return source_tree.copy_file(
            source_path.name,
            destination,
            maximum_bytes=maximum_bytes,
        )


def _hash_open_file(descriptor: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    size_bytes: int | None,
    sha256: str,
) -> Path:
    """Verify one regular file using a no-follow descriptor.

    This is useful for audits.  Runtime construction must use
    :func:`stage_verified_artifact` so it consumes the verified bytes rather than
    re-opening this user-visible path.
    """

    expected_size, expected_digest = _validate_expected(size_bytes=size_bytes, sha256=sha256)
    file_path = Path(path)
    if file_path.is_symlink():
        raise ArtifactSecurityError(f"Artifact file is a symlink: {file_path}")
    descriptor = _open_nofollow(file_path)
    actual_size, actual_digest = _hash_open_file(descriptor)
    if expected_size is not None and actual_size != expected_size:
        raise ArtifactSecurityError(
            f"Artifact file size mismatch for {file_path}: "
            f"expected {expected_size}, found {actual_size}"
        )
    if actual_digest != expected_digest:
        raise ArtifactSecurityError(
            f"Artifact file SHA-256 mismatch for {file_path}: "
            f"expected {expected_digest}, found {actual_digest}"
        )
    return file_path


def _iter_attachments(manifest: Mapping[str, Any]) -> Iterator[tuple[str, int | None, str]]:
    components = manifest.get("components", [])
    if not isinstance(components, list):
        raise ArtifactSecurityError("manifest.components must be a list")
    for index, item in enumerate(components):
        if not isinstance(item, Mapping):
            raise ArtifactSecurityError(f"manifest.components[{index}] must be an object")
        yield str(item.get("path", "")), item.get("size_bytes"), str(item.get("sha256", ""))

    policy_ref = manifest.get("policy_ref")
    if policy_ref is not None:
        if not isinstance(policy_ref, Mapping):
            raise ArtifactSecurityError("manifest.policy_ref must be an object")
        yield (
            str(policy_ref.get("path", "")),
            policy_ref.get("size_bytes"),
            str(policy_ref.get("sha256", "")),
        )

    verification = manifest.get("verification")
    if isinstance(verification, Mapping) and verification.get("report") is not None:
        report = verification.get("report")
        if not isinstance(report, Mapping):
            raise ArtifactSecurityError("manifest.verification.report must be an object")
        yield (
            str(report.get("path", "")),
            report.get("size_bytes"),
            str(report.get("sha256", "")),
        )

    attachments = manifest.get("attachments", [])
    if not isinstance(attachments, list):
        raise ArtifactSecurityError("manifest.attachments must be a list")
    for index, item in enumerate(attachments):
        if not isinstance(item, Mapping):
            raise ArtifactSecurityError(f"manifest.attachments[{index}] must be an object")
        yield str(item.get("path", "")), item.get("size_bytes"), str(item.get("sha256", ""))

    for metadata_name in ("provenance", "producer"):
        metadata = manifest.get(metadata_name)
        if not isinstance(metadata, Mapping) or metadata.get("attachments") is None:
            continue
        values = metadata.get("attachments")
        if not isinstance(values, list):
            raise ArtifactSecurityError(f"manifest.{metadata_name}.attachments must be a list")
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise ArtifactSecurityError(
                    f"manifest.{metadata_name}.attachments[{index}] must be an object"
                )
            yield (
                str(item.get("path", "")),
                item.get("size_bytes"),
                str(item.get("sha256", "")),
            )


def _attachment_table(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, tuple[Path, int | None, str]]:
    table: dict[str, tuple[Path, int | None, str]] = {}
    for raw_path, size_bytes, sha256 in _iter_attachments(manifest):
        relative = _relative_path(raw_path)
        if relative in table:
            raise ArtifactSecurityError(f"Duplicate manifest file reference: {relative}")
        expected_size, expected_digest = _validate_expected(size_bytes=size_bytes, sha256=sha256)
        table[relative] = (
            resolve_contained_path(root, relative),
            expected_size,
            expected_digest,
        )
    return table


def _read_contained_bytes(root: Path, relative: str, *, maximum: int) -> bytes:
    descriptor = _open_contained_nofollow(root, relative)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ArtifactSecurityError(
            f"Artifact component exceeds the safe inspection limit: {relative}"
        )
    return data


def _validate_onnx_dependency_closure(root: Path, manifest: Mapping[str, Any]) -> None:
    raw_components = manifest.get("components", [])
    if not isinstance(raw_components, list):
        return
    components = [item for item in raw_components if isinstance(item, Mapping)]
    onnx_models = [
        item
        for item in components
        if item.get("format") == "onnx" and item.get("serialization") == "onnx"
    ]
    external_components = [item for item in components if item.get("role") == "external_data"]
    if not onnx_models and not external_components:
        return
    if not onnx_models:
        raise ArtifactSecurityError(
            "Artifact declares ONNX external_data without an ONNX executable component."
        )

    external_paths: set[str] = set()
    for item in external_components:
        if (item.get("format"), item.get("serialization")) != (
            "onnx-external-data",
            "safe-data",
        ):
            raise ArtifactSecurityError(
                "ONNX external_data components require format='onnx-external-data' "
                "and serialization='safe-data'."
            )
        external_paths.add(_relative_path(str(item.get("path", ""))))

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ArtifactSecurityError(
            "Secure ONNX artifact inspection requires pyimgano[onnx-runtime]."
        ) from exc

    referenced_paths: set[str] = set()
    for item in onnx_models:
        model_relative = _relative_path(str(item.get("path", "")))
        model_bytes = _read_contained_bytes(
            root,
            model_relative,
            maximum=_MAX_ONNX_PROTOBUF_BYTES,
        )
        try:
            # Parse protobuf bytes directly.  The convenience ONNX loaders may
            # resolve external_data as a side effect, which is forbidden before
            # the dependency closure has been validated.
            model = onnx.ModelProto()
            model.ParseFromString(model_bytes)
            locations = external_data_locations(model)
        except Exception as exc:
            raise ArtifactSecurityError(
                f"Cannot safely inspect ONNX dependency closure for {model_relative}: {exc}"
            ) from exc

        model_parent = PurePosixPath(model_relative).parent
        for location in locations:
            try:
                safe_location = _relative_path(location)
                dependency = _relative_path(
                    model_parent.joinpath(PurePosixPath(safe_location)).as_posix()
                )
            except ArtifactSecurityError as exc:
                raise ArtifactSecurityError(
                    f"Unsafe ONNX external-data location in {model_relative}: {location!r}"
                ) from exc
            referenced_paths.add(dependency)

    missing = sorted(referenced_paths - external_paths)
    unexpected = sorted(external_paths - referenced_paths)
    if missing:
        raise ArtifactSecurityError(
            f"ONNX external-data dependencies are missing from manifest components: {missing}"
        )
    if unexpected:
        raise ArtifactSecurityError(
            f"Manifest contains unreferenced ONNX external_data components: {unexpected}"
        )


def verify_artifact_files(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Verify every manifest-listed component, policy, and report attachment."""

    root_path = _ensure_root(root)
    table = _attachment_table(root_path, manifest)
    verified: dict[str, Path] = {}
    for relative, (path, size_bytes, digest) in table.items():
        descriptor = _open_contained_nofollow(root_path, relative)
        actual_size, actual_digest = _hash_open_file(descriptor)
        if size_bytes is not None and actual_size != size_bytes:
            raise ArtifactSecurityError(
                f"Artifact file size mismatch for {path}: "
                f"expected {size_bytes}, found {actual_size}"
            )
        if actual_digest != digest:
            raise ArtifactSecurityError(
                f"Artifact file SHA-256 mismatch for {path}: "
                f"expected {digest}, found {actual_digest}"
            )
        verified[relative] = path
    _validate_onnx_dependency_closure(root_path, manifest)
    return verified


def _copy_verified_file(
    root: Path,
    relative: str,
    source: Path,
    destination: Path,
    *,
    expected_size: int | None,
    expected_digest: str,
) -> None:
    descriptor = _open_contained_nofollow(root, relative)
    before = os.fstat(descriptor)
    if expected_size is not None and before.st_size != expected_size:
        os.close(descriptor)
        raise ArtifactSecurityError(
            f"Artifact file size mismatch for {source}: expected {expected_size}, found {before.st_size}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            os.fdopen(descriptor, "rb", closefd=True) as source_handle,
            destination.open("xb") as destination_handle,
        ):
            while True:
                chunk = source_handle.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                destination_handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
            after = os.fstat(source_handle.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        destination.unlink(missing_ok=True)
        raise ArtifactSecurityError(f"Artifact file changed identity while staging: {source}")
    if expected_size is not None and total != expected_size:
        destination.unlink(missing_ok=True)
        raise ArtifactSecurityError(
            f"Artifact file size mismatch for {source}: expected {expected_size}, found {total}"
        )
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        destination.unlink(missing_ok=True)
        raise ArtifactSecurityError(
            f"Artifact file SHA-256 mismatch for {source}: "
            f"expected {expected_digest}, found {actual_digest}"
        )
    destination.chmod(0o400)


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, directories, files in os.walk(root):
        current = Path(directory)
        try:
            current.chmod(0o700)
        except OSError:
            pass
        for name in directories:
            try:
                (current / name).chmod(0o700)
            except OSError:
                pass
        for name in files:
            try:
                (current / name).chmod(0o600)
            except OSError:
                pass


@dataclass
class VerifiedArtifactStaging:
    """Process-private immutable copy of the exact bytes that passed verification."""

    root: Path
    _paths: dict[str, Path] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def paths(self) -> Mapping[str, Path]:
        return dict(self._paths)

    def path_for(self, relative_path: str | Path) -> Path:
        relative = _relative_path(relative_path)
        try:
            return self._paths[relative]
        except KeyError as exc:
            raise ArtifactSecurityError(
                f"Path was not part of the verified staging closure: {relative}"
            ) from exc

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.root.exists():
            _make_tree_writable(self.root)
            shutil.rmtree(self.root)

    def __enter__(self) -> "VerifiedArtifactStaging":
        if self._closed:
            raise ArtifactSecurityError("Verified artifact staging has already been closed.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.cleanup()


def stage_verified_artifact(
    root: str | Path, manifest: Mapping[str, Any]
) -> VerifiedArtifactStaging:
    """Copy and hash all runtime files from one open descriptor into a private tree."""

    root_path = _ensure_root(root)
    table = _attachment_table(root_path, manifest)
    staging_root = Path(tempfile.mkdtemp(prefix="pyimgano-artifact-"))
    staging_root.chmod(0o700)
    paths: dict[str, Path] = {}
    try:
        for relative, (source, size_bytes, digest) in table.items():
            destination = staging_root.joinpath(*relative.split("/"))
            _copy_verified_file(
                root_path,
                relative,
                source,
                destination,
                expected_size=size_bytes,
                expected_digest=digest,
            )
            paths[relative] = destination

        _validate_onnx_dependency_closure(staging_root, manifest)

        if manifest.get("schema_family") is not None:
            from pyimgano.artifacts.manifest import canonical_json_bytes

            manifest_path = staging_root / "artifact_manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            manifest_path.chmod(0o400)
            paths["artifact_manifest.json"] = manifest_path

        directories = [Path(directory) for directory, _, _ in os.walk(staging_root)]
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            directory.chmod(0o500)
        return VerifiedArtifactStaging(root=staging_root, _paths=paths)
    except Exception:
        _make_tree_writable(staging_root)
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


__all__ = [
    "ArtifactSecurityError",
    "SecureSourceTree",
    "VerifiedArtifactStaging",
    "copy_regular_file_nofollow",
    "resolve_contained_path",
    "stage_verified_artifact",
    "verify_artifact_files",
    "verify_file",
]
