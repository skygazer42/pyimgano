from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from pyimgano.exporting.types import CheckpointCompleteness, CheckpointContract
from pyimgano.serialization.safe_checkpoint import (
    SafeCheckpointError,
    load_safe_checkpoint,
    save_safe_checkpoint,
)

FITTED_STATE_FORMAT = "pyimgano.fitted-detector-state"
FITTED_STATE_VERSION = 1

_OPERATING_POLICY_KEYS = frozenset(
    {
        "threshold",
        "threshold_",
        "operating_threshold",
        "image_threshold",
        "pixel_threshold",
        "labels_",
        "decision_labels_",
    }
)


class StateCodecError(SafeCheckpointError):
    pass


@dataclass(frozen=True)
class StateField:
    name: str
    attribute: str | None = None
    required: bool = True
    dtypes: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()
    max_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("State field name must not be empty.")
        if name in _OPERATING_POLICY_KEYS:
            raise ValueError(
                f"Operating-policy field cannot be registered as fitted state: {name!r}"
            )
        if self.attribute is not None and str(self.attribute).strip() in _OPERATING_POLICY_KEYS:
            raise ValueError(
                f"Operating-policy attribute cannot be registered as fitted state: {self.attribute!r}"
            )
        if int(self.max_bytes) < 0:
            raise ValueError("State field max_bytes must not be negative.")
        if any(int(rank) < 0 for rank in self.ranks):
            raise ValueError("State field ranks must not be negative.")


class StateCodecProtocol(Protocol):
    codec_id: str
    codec_version: int
    state_schema_version: int
    model_names: Sequence[str]

    def encode(self, detector: Any) -> Mapping[str, Any]: ...

    def validate_state(self, state: Mapping[str, Any]) -> None: ...

    def decode(self, detector: Any, state: Mapping[str, Any]) -> None: ...


def _array_view(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    module_name = str(getattr(type(value), "__module__", ""))
    if module_name.startswith("torch"):
        detach = getattr(value, "detach", None)
        if callable(detach):
            normalized = value.detach()
            cpu = getattr(normalized, "cpu", None)
            if callable(cpu):
                normalized = cpu()
            numpy_fn = getattr(normalized, "numpy", None)
            if callable(numpy_fn):
                return np.asarray(numpy_fn())
    return None


def _reject_operating_policy_state(value: Any, *, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in _OPERATING_POLICY_KEYS:
                raise StateCodecError(
                    f"Operating-policy value must not be serialized as computational state: "
                    f"{path}.{key}"
                )
            _reject_operating_policy_state(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_operating_policy_state(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class MappingStateCodec:
    """Explicit allowlisted attribute codec for simple detector families."""

    codec_id: str
    codec_version: int
    state_schema_version: int
    model_names: tuple[str, ...]
    fields: tuple[StateField, ...]

    def __post_init__(self) -> None:
        if not str(self.codec_id).strip():
            raise ValueError("codec_id must not be empty.")
        if int(self.codec_version) < 1 or int(self.state_schema_version) < 1:
            raise ValueError("Codec versions must be positive integers.")
        if not self.model_names or any(not str(name).strip() for name in self.model_names):
            raise ValueError("A state codec must bind at least one model name.")
        field_names = [str(field.name) for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("State codec field names must be unique.")

    def encode(self, detector: Any) -> Mapping[str, Any]:
        state: dict[str, Any] = {}
        for field in self.fields:
            attribute = str(field.attribute or field.name)
            if not hasattr(detector, attribute):
                if field.required:
                    raise StateCodecError(
                        f"Required fitted-state attribute is missing: {attribute!r}."
                    )
                continue
            state[str(field.name)] = getattr(detector, attribute)
        self.validate_state(state)
        return state

    def validate_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise StateCodecError("Fitted state must be a mapping.")
        by_name = {str(field.name): field for field in self.fields}
        extra = sorted(set(str(key) for key in state) - set(by_name))
        if extra:
            raise StateCodecError(f"Fitted state contains unregistered fields: {extra!r}.")
        missing = sorted(
            name for name, field in by_name.items() if field.required and name not in state
        )
        if missing:
            raise StateCodecError(f"Fitted state is missing required fields: {missing!r}.")

        _reject_operating_policy_state(state)
        for name, value in state.items():
            field = by_name[str(name)]
            array = _array_view(value)
            if array is None:
                if field.dtypes or field.ranks:
                    raise StateCodecError(f"Fitted-state field {name!r} must be an array/tensor.")
                continue
            if array.dtype.hasobject or array.dtype.kind in {"O", "V", "U", "S"}:
                raise StateCodecError(
                    f"Fitted-state field {name!r} has unsupported dtype {array.dtype!s}."
                )
            if field.dtypes and str(array.dtype) not in set(field.dtypes):
                raise StateCodecError(
                    f"Fitted-state field {name!r} has dtype {array.dtype!s}; "
                    f"expected one of {sorted(field.dtypes)!r}."
                )
            if field.ranks and int(array.ndim) not in set(int(rank) for rank in field.ranks):
                raise StateCodecError(
                    f"Fitted-state field {name!r} has rank {array.ndim}; "
                    f"expected one of {sorted(field.ranks)!r}."
                )
            if int(array.nbytes) > int(field.max_bytes):
                raise StateCodecError(f"Fitted-state field {name!r} exceeds its byte limit.")

    def decode(self, detector: Any, state: Mapping[str, Any]) -> None:
        self.validate_state(state)
        for field in self.fields:
            name = str(field.name)
            if name in state:
                setattr(detector, str(field.attribute or name), state[name])


class StateCodecRegistry:
    def __init__(self) -> None:
        self._by_id: dict[tuple[str, int], StateCodecProtocol] = {}
        self._latest_by_id: dict[str, int] = {}
        self._by_model: dict[str, tuple[str, int]] = {}

    def register(self, codec: StateCodecProtocol, *, overwrite: bool = False) -> StateCodecProtocol:
        codec_id = str(getattr(codec, "codec_id", "")).strip()
        version = int(getattr(codec, "codec_version", 0))
        schema_version = int(getattr(codec, "state_schema_version", 0))
        model_names = tuple(str(name).strip() for name in getattr(codec, "model_names", ()))
        if not codec_id or version < 1 or schema_version < 1 or not model_names:
            raise ValueError("State codecs require id/version/schema and at least one model name.")
        key = (codec_id, version)
        occupied_models = [name for name in model_names if name in self._by_model]
        if (key in self._by_id or occupied_models) and not overwrite:
            raise ValueError(
                f"State codec registration conflicts with existing entries: {codec_id!r} "
                f"v{version}, models={occupied_models!r}."
            )
        if overwrite:
            for model_name in occupied_models:
                self._by_model.pop(model_name, None)
        self._by_id[key] = codec
        self._latest_by_id[codec_id] = max(version, self._latest_by_id.get(codec_id, 0))
        for model_name in model_names:
            if not model_name:
                raise ValueError("State codec model names must not be empty.")
            self._by_model[model_name] = key
        return codec

    def get(self, codec_id: str, codec_version: int | None = None) -> StateCodecProtocol:
        normalized = str(codec_id).strip()
        version = (
            int(codec_version)
            if codec_version is not None
            else self._latest_by_id.get(normalized, 0)
        )
        try:
            return self._by_id[(normalized, version)]
        except KeyError as exc:
            raise KeyError(f"State codec not registered: {normalized!r} v{version}.") from exc

    def for_model(self, model_name: str) -> StateCodecProtocol:
        normalized = str(model_name).strip()
        try:
            codec_id, version = self._by_model[normalized]
        except KeyError as exc:
            raise KeyError(
                f"No safe fitted-state codec registered for model {normalized!r}."
            ) from exc
        return self.get(codec_id, version)

    def clear(self) -> None:
        self._by_id.clear()
        self._latest_by_id.clear()
        self._by_model.clear()


STATE_CODEC_REGISTRY = StateCodecRegistry()


def register_state_codec(
    codec: StateCodecProtocol, *, overwrite: bool = False
) -> StateCodecProtocol:
    return STATE_CODEC_REGISTRY.register(codec, overwrite=overwrite)


def get_state_codec(codec_id: str, codec_version: int | None = None) -> StateCodecProtocol:
    return STATE_CODEC_REGISTRY.get(codec_id, codec_version)


def get_state_codec_for_model(model_name: str) -> StateCodecProtocol:
    return STATE_CODEC_REGISTRY.for_model(model_name)


@dataclass(frozen=True)
class FittedStateInfo:
    model_name: str
    codec_id: str
    codec_version: int
    state_schema_version: int
    completeness: CheckpointCompleteness


def save_fitted_state(
    detector: Any,
    path: str | Path,
    *,
    model_name: str,
    checkpoint_contract: CheckpointContract,
    codec_id: str | None = None,
) -> Path:
    if not checkpoint_contract.strict_exportable:
        raise StateCodecError(
            "Strict native export requires a complete, verified checkpoint contract; "
            f"got completeness={checkpoint_contract.completeness!s}."
        )
    codec = (
        get_state_codec(str(codec_id))
        if codec_id is not None
        else get_state_codec_for_model(model_name)
    )
    if str(model_name) not in {str(name) for name in codec.model_names}:
        raise StateCodecError(
            f"State codec {codec.codec_id!r} is not bound to model {model_name!r}."
        )
    state = dict(codec.encode(detector))
    codec.validate_state(state)
    _reject_operating_policy_state(state)
    return save_safe_checkpoint(
        {
            "format": FITTED_STATE_FORMAT,
            "version": FITTED_STATE_VERSION,
            "model_name": str(model_name),
            "codec_id": str(codec.codec_id),
            "codec_version": int(codec.codec_version),
            "state_schema_version": int(codec.state_schema_version),
            "completeness": CheckpointCompleteness.COMPLETE.value,
            "source_checkpoint": checkpoint_contract.to_dict(),
            "state": state,
        },
        path,
    )


def inspect_fitted_state(path: str | Path) -> FittedStateInfo:
    payload = load_safe_checkpoint(path)
    if payload.get("format") != FITTED_STATE_FORMAT:
        raise StateCodecError("Checkpoint is not a fitted-detector state archive.")
    if int(payload.get("version", -1)) != FITTED_STATE_VERSION:
        raise StateCodecError("Unsupported fitted-detector state archive version.")
    try:
        completeness = CheckpointCompleteness(str(payload.get("completeness", "unknown")))
    except ValueError as exc:
        raise StateCodecError("Invalid fitted-state completeness value.") from exc
    return FittedStateInfo(
        model_name=str(payload.get("model_name", "")),
        codec_id=str(payload.get("codec_id", "")),
        codec_version=int(payload.get("codec_version", 0)),
        state_schema_version=int(payload.get("state_schema_version", 0)),
        completeness=completeness,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint_contract(path: str | Path) -> CheckpointContract:
    """Inspect persisted evidence without treating loadability as completeness."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    size_bytes = int(checkpoint.stat().st_size)
    sha256 = _sha256_file(checkpoint)
    try:
        payload = load_safe_checkpoint(checkpoint)
    except SafeCheckpointError:
        # Unknown torch/joblib/detector-specific files carry no certified codec
        # evidence.  Loading them elsewhere must not promote this result.
        return CheckpointContract(
            completeness=CheckpointCompleteness.UNKNOWN,
            size_bytes=size_bytes,
            sha256=sha256,
            roundtrip_verified=False,
        )

    if payload.get("format") == FITTED_STATE_FORMAT:
        source = payload.get("source_checkpoint")
        if not isinstance(source, Mapping):
            raise StateCodecError("Fitted-state archive is missing source checkpoint evidence.")
        contract = CheckpointContract.from_mapping(source)
        return replace(contract, size_bytes=size_bytes, sha256=sha256)

    # Legacy safe detector-state and arbitrary structured checkpoints remain
    # unknown unless a registered adapter separately certified the run.
    return CheckpointContract(
        completeness=CheckpointCompleteness.UNKNOWN,
        size_bytes=size_bytes,
        sha256=sha256,
        roundtrip_verified=False,
    )


def load_fitted_state(
    detector: Any,
    path: str | Path,
    *,
    expected_model_name: str | None = None,
) -> FittedStateInfo:
    payload = load_safe_checkpoint(path)
    info = inspect_fitted_state(path)
    if info.completeness is not CheckpointCompleteness.COMPLETE:
        raise StateCodecError(
            "Fitted-state archive is not certified complete; loading it cannot upgrade completeness."
        )
    if expected_model_name is not None and info.model_name != str(expected_model_name):
        raise StateCodecError(
            f"Fitted-state model mismatch: loaded={info.model_name!r}, "
            f"expected={str(expected_model_name)!r}."
        )
    codec = get_state_codec(info.codec_id, info.codec_version)
    if int(codec.state_schema_version) != info.state_schema_version:
        raise StateCodecError("Fitted-state schema version does not match the registered codec.")
    if info.model_name not in {str(name) for name in codec.model_names}:
        raise StateCodecError("Fitted-state codec/model binding is invalid.")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise StateCodecError("Fitted-state archive is missing its state mapping.")
    codec.validate_state(state)
    _reject_operating_policy_state(state)
    codec.decode(detector, state)
    return info


__all__ = [
    "FITTED_STATE_FORMAT",
    "FITTED_STATE_VERSION",
    "FittedStateInfo",
    "MappingStateCodec",
    "STATE_CODEC_REGISTRY",
    "StateCodecError",
    "StateCodecProtocol",
    "StateCodecRegistry",
    "StateField",
    "get_state_codec",
    "get_state_codec_for_model",
    "inspect_fitted_state",
    "inspect_checkpoint_contract",
    "load_fitted_state",
    "register_state_codec",
    "save_fitted_state",
]
