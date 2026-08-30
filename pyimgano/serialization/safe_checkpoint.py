from __future__ import annotations

"""Non-executable checkpoint archives for structured model state.

The format is a ZIP container with a JSON manifest and ``.npy`` array members.
It intentionally supports only a small allowlist of value types and always
loads NumPy arrays with ``allow_pickle=False``.
"""

import base64
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FORMAT_NAME = "pyimgano.safe-checkpoint"
FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ARRAY_COUNT = 100_000
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024


class SafeCheckpointError(ValueError):
    """Raised when a safe checkpoint is malformed or unsupported."""


def _as_numpy_array(value: Any) -> tuple[np.ndarray, bool] | None:
    if isinstance(value, np.ndarray):
        return np.asarray(value), False

    module_name = str(getattr(type(value), "__module__", ""))
    if module_name.startswith("torch"):
        detach = getattr(value, "detach", None)
        cpu = getattr(value, "cpu", None)
        numpy = getattr(value, "numpy", None)
        if callable(detach) and callable(cpu):
            normalized = value.detach().cpu()
            numpy = getattr(normalized, "numpy", None)
            if callable(numpy):
                return np.asarray(numpy()), True
    return None


def _encode_value(value: Any, arrays: dict[str, bytes], *, depth: int = 0) -> Any:
    if depth > 100:
        raise SafeCheckpointError("Checkpoint payload nesting exceeds the supported limit.")

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__type__": "float", "value": repr(value)}
    if isinstance(value, np.generic):
        return _encode_value(value.item(), arrays, depth=depth + 1)
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }

    array_payload = _as_numpy_array(value)
    if array_payload is not None:
        array, was_tensor = array_payload
        if array.dtype.hasobject:
            raise SafeCheckpointError("Object-dtype arrays are not supported in safe checkpoints.")
        name = f"arrays/{len(arrays):08d}.npy"
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
        arrays[name] = buffer.getvalue()
        return {
            "__type__": "torch_tensor" if was_tensor else "ndarray",
            "name": name,
        }

    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SafeCheckpointError("Checkpoint mappings must use string keys.")
            encoded[key] = _encode_value(item, arrays, depth=depth + 1)
        return {"__type__": "mapping", "items": encoded}
    if isinstance(value, (list, tuple)):
        return {
            "__type__": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode_value(item, arrays, depth=depth + 1) for item in value],
        }

    raise SafeCheckpointError(
        "Unsupported value in safe checkpoint: " f"{type(value).__module__}.{type(value).__name__}"
    )


def _decode_value(value: Any, arrays: Mapping[str, np.ndarray], *, depth: int = 0) -> Any:
    if depth > 100:
        raise SafeCheckpointError("Checkpoint payload nesting exceeds the supported limit.")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, dict):
        raise SafeCheckpointError("Invalid value node in safe checkpoint manifest.")

    kind = value.get("__type__")
    if kind == "float":
        raw = str(value.get("value", ""))
        allowed = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
        if raw not in allowed:
            raise SafeCheckpointError("Invalid non-finite float marker.")
        return allowed[raw]
    if kind == "path":
        return Path(str(value.get("value", "")))
    if kind == "bytes":
        try:
            return base64.b64decode(str(value.get("value", "")), validate=True)
        except Exception as exc:
            raise SafeCheckpointError("Invalid base64 bytes value.") from exc
    if kind in {"ndarray", "torch_tensor"}:
        name = str(value.get("name", ""))
        if name not in arrays:
            raise SafeCheckpointError(f"Checkpoint array is missing: {name}")
        array = np.asarray(arrays[name])
        if kind == "torch_tensor":
            try:
                import torch
            except Exception:
                return array
            return torch.from_numpy(np.array(array, copy=True))
        return array
    if kind == "mapping":
        items = value.get("items")
        if not isinstance(items, dict):
            raise SafeCheckpointError("Invalid mapping node in safe checkpoint manifest.")
        return {
            str(key): _decode_value(item, arrays, depth=depth + 1) for key, item in items.items()
        }
    if kind in {"list", "tuple"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise SafeCheckpointError("Invalid sequence node in safe checkpoint manifest.")
        decoded = [_decode_value(item, arrays, depth=depth + 1) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    raise SafeCheckpointError(f"Unsupported safe checkpoint node type: {kind!r}")


def save_safe_checkpoint(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a structured, non-executable checkpoint atomically."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, bytes] = {}
    encoded_root = _encode_value(dict(payload), arrays)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "root": encoded_root,
        "arrays": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in arrays.items()
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent, delete=False
        ) as temp:
            temp_path = Path(temp.name)
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(_MANIFEST_NAME, manifest_bytes)
            for name, data in arrays.items():
                archive.writestr(name, data)
        os.replace(temp_path, out_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return out_path


def load_safe_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a structured checkpoint without executing object code."""

    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")

    try:
        with zipfile.ZipFile(in_path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SafeCheckpointError("Checkpoint archive contains duplicate members.")
            if _MANIFEST_NAME not in names:
                raise SafeCheckpointError("Checkpoint manifest is missing.")
            if len(infos) - 1 > _MAX_ARRAY_COUNT:
                raise SafeCheckpointError("Checkpoint contains too many array members.")
            total_size = sum(int(info.file_size) for info in infos)
            if total_size > _MAX_UNCOMPRESSED_BYTES:
                raise SafeCheckpointError("Checkpoint exceeds the supported uncompressed size.")

            manifest_info = archive.getinfo(_MANIFEST_NAME)
            if int(manifest_info.file_size) > _MAX_MANIFEST_BYTES:
                raise SafeCheckpointError("Checkpoint manifest exceeds the supported size.")
            manifest = json.loads(archive.read(_MANIFEST_NAME).decode("utf-8"))
            if not isinstance(manifest, dict):
                raise SafeCheckpointError("Checkpoint manifest must be a JSON object.")
            if manifest.get("format") != FORMAT_NAME:
                raise SafeCheckpointError("Unsupported checkpoint format marker.")
            if int(manifest.get("version", -1)) != FORMAT_VERSION:
                raise SafeCheckpointError("Unsupported checkpoint format version.")

            array_meta = manifest.get("arrays")
            if not isinstance(array_meta, dict):
                raise SafeCheckpointError("Checkpoint array manifest is invalid.")
            expected_names = {_MANIFEST_NAME, *(str(name) for name in array_meta)}
            if set(names) != expected_names:
                raise SafeCheckpointError("Checkpoint archive members do not match its manifest.")

            arrays: dict[str, np.ndarray] = {}
            for name, meta in array_meta.items():
                if not isinstance(name, str) or not name.startswith("arrays/"):
                    raise SafeCheckpointError("Checkpoint array member name is invalid.")
                if not isinstance(meta, dict):
                    raise SafeCheckpointError("Checkpoint array metadata is invalid.")
                data = archive.read(name)
                if len(data) != int(meta.get("size", -1)):
                    raise SafeCheckpointError(f"Checkpoint array size mismatch: {name}")
                digest = hashlib.sha256(data).hexdigest()
                if digest != str(meta.get("sha256", "")):
                    raise SafeCheckpointError(f"Checkpoint array digest mismatch: {name}")
                try:
                    arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)
                except Exception as exc:
                    raise SafeCheckpointError(f"Invalid checkpoint array: {name}") from exc
    except zipfile.BadZipFile as exc:
        raise SafeCheckpointError("Checkpoint is not a valid safe archive.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeCheckpointError("Checkpoint manifest is not valid UTF-8 JSON.") from exc

    decoded = _decode_value(manifest.get("root"), arrays)
    if not isinstance(decoded, dict):
        raise SafeCheckpointError("Checkpoint root payload must be a mapping.")
    return decoded


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SafeCheckpointError",
    "load_safe_checkpoint",
    "save_safe_checkpoint",
]
