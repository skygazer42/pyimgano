from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import numpy as np

from pyimgano.artifacts.compatibility import (
    current_platform_tag,
    onnxruntime_requirement_for_graph,
)
from pyimgano.artifacts.onnx_contract import normalize_onnx_import_contract
from pyimgano.artifacts.onnx_external_data import external_data_locations
from pyimgano.artifacts.onnx_graph import validate_onnx_model_contract
from pyimgano.artifacts.security import SecureSourceTree

_MAX_ONNX_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Policy must be a JSON object/dict: {source}")
    return dict(payload)


def _default_score_only_policy() -> dict[str, Any]:
    return {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "postprocess": {
            "schema_version": 1,
            "threshold_scope": "image",
            "image_threshold": {
                "threshold": None,
                "score_order": "higher_is_more_anomalous",
                "provenance": None,
            },
            "pixel_threshold": {
                "enabled": False,
                "strategy": "fixed",
                "threshold": None,
                "normal_quantile": None,
            },
            "map_postprocess": None,
            "review_policy": {
                "review_on": ["anomalous"],
                "confidence_gate_enabled": False,
                "reject_confidence_below": None,
                "reject_label": None,
            },
            "label_encoding": {"normal": 0, "anomalous": 1},
        },
        "defects": {"enabled": False},
    }


def _validate_external_location(location: str) -> str:
    if not location or location != location.strip():
        raise ValueError(
            f"Unsafe ONNX external-data location; surrounding whitespace is forbidden: {location!r}"
        )
    if unicodedata.normalize("NFC", location) != location:
        raise ValueError(
            f"Unsafe ONNX external-data location; NFC normalization is required: {location!r}"
        )
    if "\x00" in location or "\\" in location or "//" in location or location.endswith("/"):
        raise ValueError(f"Unsafe ONNX external-data location: {location!r}")
    posix = PurePosixPath(location)
    windows = PureWindowsPath(location)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"Unsafe ONNX external-data location: {location!r}")
    for part in location.split("/"):
        if part in {"", ".", ".."}:
            raise ValueError(f"Unsafe ONNX external-data location: {location!r}")
    return location


def _runtime_smoke(model_path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ONNX import runtime verification requires pyimgano[onnx-runtime]."
        ) from exc

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_contract = dict(contract["input"])
    height, width = (int(v) for v in input_contract["size"])
    channels = 1 if input_contract["color_space"] == "GRAY" else 3
    shape = (
        (1, channels, height, width)
        if input_contract["layout"] == "NCHW"
        else (1, height, width, channels)
    )
    sample = np.zeros(shape, dtype=np.dtype(str(input_contract["dtype"])))
    output_names = [str(item["name"]) for item in dict(contract["outputs"]).values()]
    values = session.run(output_names, {str(input_contract["name"]): sample})
    outputs: list[dict[str, Any]] = []
    for name, value in zip(output_names, values):
        array = np.asarray(value)
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            raise ValueError(f"ONNX runtime smoke produced non-finite values for output {name!r}.")
        outputs.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype)})
    return {
        "schema_family": "pyimgano-artifact-verification",
        "schema_version": 1,
        "level": "runtime_smoke",
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "input_shape": list(shape),
        "outputs": outputs,
    }


def _component(
    path: Path, *, root: Path, role: str, format: str, serialization: str
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "format": format,
        "serialization": serialization,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _atomic_publish(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact output already exists: {destination}")
    backup = None
    try:
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
            os.replace(destination, backup)
        os.replace(staging, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def import_onnx(
    model: str | Path,
    *,
    contract: str | Path | Mapping[str, Any] | None,
    out: str | Path,
    policy: str | Path | Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if contract is None:
        raise ValueError(
            "A versioned ONNX import contract is required; tensor shapes cannot define anomaly semantics."
        )
    normalized_contract = normalize_onnx_import_contract(contract)
    source = Path(model)
    if source.is_symlink():
        raise ValueError(f"ONNX model source must not be a symlink: {source}")

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("ONNX import requires pyimgano[onnx-runtime].") from exc
    policy_payload = (
        dict(policy)
        if isinstance(policy, Mapping)
        else (_read_json_mapping(policy) if policy is not None else _default_score_only_policy())
    )
    from pyimgano.artifacts import validate_artifact_policy, write_artifact_manifest

    policy_payload = validate_artifact_policy(policy_payload)

    with SecureSourceTree(source.parent) as source_tree:
        destination = Path(out).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent))
        )
        try:
            model_dir = staging / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            target_model = model_dir / "model.onnx"
            source_tree.copy_file(
                source.name,
                target_model,
                maximum_bytes=_MAX_ONNX_BYTES,
            )

            # Parse and validate only the private bytes copied from the stable
            # source descriptor.  Source paths are never reopened after checks.
            graph = onnx.load_model(str(target_model), load_external_data=False)
            graph_info = validate_onnx_model_contract(
                graph,
                input_contract=normalized_contract["input"],
                output_contract=normalized_contract["outputs"],
                onnx_module=onnx,
            )
            locations = [
                _validate_external_location(location) for location in external_data_locations(graph)
            ]
            if any(location.casefold() == "model.onnx" for location in locations):
                raise ValueError(
                    "ONNX external data must not overwrite the artifact model entrypoint."
                )

            external_targets: list[Path] = []
            for location in locations:
                target = model_dir.joinpath(*location.split("/"))
                source_tree.copy_file(location, target)
                external_targets.append(target)

            try:
                # The path-based checker can now resolve only the private,
                # canonical external-data closure copied above.
                onnx.checker.check_model(str(target_model), full_check=True)
            except Exception as exc:
                raise ValueError(f"ONNX graph failed schema/operator validation: {exc}") from exc

            policy_path = staging / "infer_config.json"
            policy_path.write_text(
                json.dumps(
                    policy_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            verification_payload = _runtime_smoke(target_model, normalized_contract)
            verification_dir = staging / "verification"
            verification_dir.mkdir(parents=True, exist_ok=True)
            verification_path = verification_dir / "runtime_smoke.json"
            verification_path.write_text(
                json.dumps(
                    verification_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            components = [
                _component(
                    target_model,
                    root=staging,
                    role="runtime_model",
                    format="onnx",
                    serialization="onnx",
                )
            ]
            components.extend(
                _component(
                    target,
                    root=staging,
                    role="external_data",
                    format="onnx-external-data",
                    serialization="safe-data",
                )
                for target in external_targets
            )
            manifest_payload: dict[str, Any] = {
                "schema_family": "pyimgano-artifact",
                "schema_version": 1,
                "layout": "single_graph",
                "runtime": {
                    "backend": "onnxruntime",
                    "entrypoint": "model/model.onnx",
                    "allowed_providers": [{"name": "CPUExecutionProvider", "options": {}}],
                    "verified_providers": [{"name": "CPUExecutionProvider", "options": {}}],
                },
                "input_contract": dict(normalized_contract["input"]),
                "output_contract": dict(normalized_contract["outputs"]),
                "components": components,
                "policy_ref": {
                    "path": "infer_config.json",
                    "sha256": _sha256(policy_path),
                },
                "compatibility": {
                    "pyimgano": ">=0.10,<0.11",
                    "python": ">=3.9,<3.13",
                    "platforms": [current_platform_tag()],
                    "runtime_versions": {
                        "onnxruntime": onnxruntime_requirement_for_graph(
                            ir_version=int(graph.ir_version),
                            default_opset=int(graph_info.default_opset),
                        )
                    },
                    "onnx_ir": int(graph.ir_version),
                    "onnx_opset": int(graph_info.default_opset),
                },
                "verification": {
                    "level": "runtime_smoke",
                    "report": {
                        "path": "verification/runtime_smoke.json",
                        "size_bytes": int(verification_path.stat().st_size),
                        "sha256": _sha256(verification_path),
                    },
                },
            }
            manifest_path = write_artifact_manifest(
                staging / "artifact_manifest.json",
                manifest_payload,
                policy=policy_payload,
            )
            from pyimgano.artifacts import load_artifact_manifest

            manifest = load_artifact_manifest(manifest_path)
            _atomic_publish(staging, destination, overwrite=bool(overwrite))
            return {
                "artifact_root": str(destination),
                "manifest": str(destination / "artifact_manifest.json"),
                "artifact_id": manifest.get("artifact_id"),
                "runtime_id": manifest.get("runtime_id"),
                "policy_id": manifest.get("policy_id"),
                "verification_level": "runtime_smoke",
            }
        finally:
            if staging.exists():
                shutil.rmtree(staging)


__all__ = ["import_onnx"]
