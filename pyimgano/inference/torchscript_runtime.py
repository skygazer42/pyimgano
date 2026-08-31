from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.inference.artifact_runtime import (
    ArtifactRuntimeError,
    normalize_map_output,
    normalize_score_output,
    prepare_image_batch,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _select_torch_output(output: Any, contract: Mapping[str, Any], *, default_index: int) -> Any:
    name = str(contract.get("name", "")).strip()
    if isinstance(output, Mapping):
        if name not in output:
            raise ArtifactRuntimeError(
                f"TorchScript output mapping has no key {name!r}; keys={sorted(output)}"
            )
        return output[name]
    if isinstance(output, (tuple, list)):
        index = int(contract.get("output_index", contract.get("index", default_index)))
        if index < 0 or index >= len(output):
            raise ArtifactRuntimeError(
                f"TorchScript output index {index} is outside tuple length {len(output)}."
            )
        return output[index]
    if default_index != 0:
        raise ArtifactRuntimeError("TorchScript returned one tensor but a map output was declared.")
    return output


def _torch_provider_key(spec: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": str(spec["name"]), "options": dict(spec.get("options", {}))},
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_torch_provider_specs(value: Any, *, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, Mapping)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ArtifactRuntimeError(f"{field} must be a provider spec or list of provider specs.")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, str):
            raw_name, options = raw, {}
        elif isinstance(raw, Mapping):
            raw_name = str(raw.get("name", ""))
            raw_options = raw.get("options", {})
            if not isinstance(raw_options, Mapping):
                raise ArtifactRuntimeError(f"{field} provider options must be a mapping.")
            options = dict(raw_options)
        else:
            raise ArtifactRuntimeError(f"{field} entries must be strings or objects.")
        alias = raw_name.strip().upper()
        name = {"CPU": "CPU", "CUDA": "CUDA", "GPU": "CUDA"}.get(alias)
        if name is None:
            raise ArtifactRuntimeError(f"{field} contains unsupported Torch provider {raw_name!r}.")
        unknown = sorted(set(options) - ({"device_id"} if name == "CUDA" else set()))
        if unknown:
            raise ArtifactRuntimeError(
                f"{field} provider {name!r} contains unsupported option(s): {unknown!r}."
            )
        normalized_options: dict[str, Any] = {}
        if "device_id" in options:
            device_id = options["device_id"]
            if (
                isinstance(device_id, bool)
                or not isinstance(device_id, (str, int))
                or not str(device_id).isdigit()
            ):
                raise ArtifactRuntimeError(f"{field} CUDA device_id must be an integer >= 0.")
            normalized_options["device_id"] = int(device_id)
        spec = {"name": name, "options": normalized_options}
        key = _torch_provider_key(spec)
        if key in seen:
            raise ArtifactRuntimeError(f"{field} contains a duplicate provider spec: {spec!r}.")
        seen.add(key)
        specs.append(spec)
    return specs


def _torch_device_spec(device: str) -> dict[str, Any]:
    value = str(device).strip().lower()
    if value == "cpu":
        return {"name": "CPU", "options": {}}
    match = re.fullmatch(r"(?:cuda|gpu)(?::([0-9]+))?", value)
    if match is None:
        raise ArtifactRuntimeError(f"Unsupported TorchScript device override: {device!r}.")
    options = {} if match.group(1) is None else {"device_id": int(match.group(1))}
    return {"name": "CUDA", "options": options}


def resolve_torchscript_device(
    torch_module: Any,
    *,
    allowed: Any,
    verified: Any,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one exact allowed-and-verified Torch device without fallback."""

    allowed_specs = _normalize_torch_provider_specs(
        allowed,
        field="runtime.allowed_providers",
    )
    verified_specs = _normalize_torch_provider_specs(
        verified,
        field="runtime.verified_providers",
    )
    if not allowed_specs or not verified_specs:
        raise ArtifactRuntimeError(
            "TorchScript artifacts require non-empty allowed_providers and verified_providers."
        )
    allowed_keys = {_torch_provider_key(item) for item in allowed_specs}
    verified_keys = {_torch_provider_key(item) for item in verified_specs}
    if not verified_keys.issubset(allowed_keys):
        raise ArtifactRuntimeError(
            "runtime.verified_providers must be an exact subset of allowed_providers."
        )

    if device is not None:
        selected = _torch_device_spec(device)
        key = _torch_provider_key(selected)
        if key not in allowed_keys:
            raise ArtifactRuntimeError(
                f"TorchScript device {device!r} is not allowed by the artifact."
            )
        if key not in verified_keys:
            raise ArtifactRuntimeError(
                f"TorchScript device {device!r} is not release-verified by the artifact."
            )
    else:
        candidates = [item for item in allowed_specs if _torch_provider_key(item) in verified_keys]
        if not candidates:
            raise ArtifactRuntimeError(
                "TorchScript artifact has no allowed-and-verified device provider."
            )
        selected = candidates[0]

    if selected["name"] == "CPU":
        return "cpu", selected
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        raise ArtifactRuntimeError("Artifact-selected TorchScript CUDA device is unavailable.")
    device_id = selected["options"].get("device_id")
    if device_id is not None:
        device_count = getattr(cuda, "device_count", None)
        if callable(device_count) and int(device_count()) <= int(device_id):
            raise ArtifactRuntimeError(
                f"Artifact-selected TorchScript CUDA device {device_id} is unavailable."
            )
        return f"cuda:{int(device_id)}", selected
    return "cuda", selected


def _torch_output_numpy(value: Any, *, field: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise ArtifactRuntimeError(f"{field} must be a Torch tensor.")
    normalized = detach()
    to = getattr(normalized, "to", None)
    if callable(to):
        normalized = to("cpu")
    numpy_fn = getattr(normalized, "numpy", None)
    if not callable(numpy_fn):
        raise ArtifactRuntimeError(f"{field} cannot be converted to a NumPy tensor.")
    array = np.asarray(numpy_fn())
    if array.dtype.kind != "f":
        raise ArtifactRuntimeError(f"{field} must be floating point; got {array.dtype!s}.")
    return array


class TorchscriptArtifactRuntime:
    """Manifest-driven TorchScript full-detector runtime.

    TorchScript deserialization is executable. A path is loaded only when the
    caller explicitly passes ``trust_checkpoint=True`` after provenance review.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_contract: Mapping[str, Any],
        output_contract: Mapping[str, Any],
        allowed_providers: Any = None,
        verified_providers: Any = None,
        device: str | None = None,
        trust_checkpoint: bool = False,
        torch_module: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_contract = dict(input_contract)
        self.output_contract = dict(output_contract)
        if model is None and trust_checkpoint is not True:
            raise ArtifactRuntimeError(
                "TorchScript deserialization requires trust_checkpoint=True after "
                "verifying artifact provenance."
            )
        if torch_module is None:
            from pyimgano.utils.optional_deps import require

            torch_module = require("torch", extra="torch", purpose="loading a TorchScript artifact")
        self.torch = torch_module
        device_name, selected_provider = resolve_torchscript_device(
            torch_module,
            allowed=allowed_providers,
            verified=verified_providers,
            device=device,
        )
        self.device = torch_module.device(device_name)
        if model is None:
            from pyimgano.utils.torchscript_safe import load_module

            model = load_module(
                self.model_path,
                map_location=self.device,
                trusted=True,
            )
        self.model = model.to(self.device) if callable(getattr(model, "to", None)) else model
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()
        # The serialized TorchScript public schema does not expose reliable
        # cross-version tensor shapes. Export parity probes and the strict first
        # (and every subsequent) output normalization below enforce the contract.
        self.runtime_info = {
            "backend": "torchscript",
            "device": str(self.device),
            "providers": [str(selected_provider["name"])],
            "selected_provider": str(selected_provider["name"]),
        }

    def score_and_maps(
        self, inputs: Sequence[Any], *, include_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]:
        items = list(inputs)
        if not items:
            return np.zeros((0,), dtype=np.float32), None
        batch, source_shapes = prepare_image_batch(items, self.input_contract)
        tensor = self.torch.from_numpy(batch).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
        score_contract = _mapping(self.output_contract.get("score"))
        score_tensor = _select_torch_output(output, score_contract, default_index=0)
        score_value = _torch_output_numpy(score_tensor, field="TorchScript score output")
        scores = normalize_score_output(score_value, score_contract, batch_size=len(items))

        maps = None
        map_contract = _mapping(self.output_contract.get("anomaly_map"))
        if include_maps and map_contract:
            map_tensor = _select_torch_output(output, map_contract, default_index=1)
            map_value = _torch_output_numpy(map_tensor, field="TorchScript anomaly-map output")
            maps = normalize_map_output(
                map_value,
                map_contract,
                batch_size=len(items),
                source_shapes=source_shapes,
            )
        return scores, maps

    def decision_function(self, inputs: Sequence[Any]) -> np.ndarray:
        return self.score_and_maps(inputs, include_maps=False)[0]

    def predict_anomaly_map(self, inputs: Sequence[Any]) -> np.ndarray | list[np.ndarray]:
        _scores, maps = self.score_and_maps(inputs, include_maps=True)
        if maps is None:
            raise ArtifactRuntimeError("TorchScript artifact has no declared anomaly-map output.")
        return maps


# Public spelling follows the backend's official "TorchScript" capitalization.
TorchScriptArtifactRuntime = TorchscriptArtifactRuntime


__all__ = [
    "TorchScriptArtifactRuntime",
    "TorchscriptArtifactRuntime",
    "resolve_torchscript_device",
]
