from __future__ import annotations

import copy

import pytest

from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
from pyimgano.inference.native_runtime_contract import resolve_native_device

_CPU = {"name": "CPU", "options": {}}
_CUDA = {"name": "CUDA", "options": {}}


@pytest.mark.parametrize(
    ("requested", "expected_device", "expected_spec"),
    [
        ("cpu", "cpu", _CPU),
        (" CPU:0 ", "cpu", _CPU),
        ("cuda", "cuda", _CUDA),
        ("GPU", "cuda", _CUDA),
        ("cuda:3", "cuda:3", {"name": "CUDA", "options": {"device_id": 3}}),
    ],
)
def test_native_device_override_normalizes_to_canonical_exact_spec(
    requested: str,
    expected_device: str,
    expected_spec: dict,
) -> None:
    device, spec = resolve_native_device(
        allowed=[expected_spec],
        verified=[expected_spec],
        device=requested,
    )

    assert device == expected_device
    assert spec == expected_spec


def test_native_default_uses_first_allowed_spec_that_is_also_verified() -> None:
    device, spec = resolve_native_device(
        allowed=[
            {"name": "CUDA", "options": {"device_id": 2}},
            _CPU,
            _CUDA,
        ],
        verified=[_CPU, _CUDA],
    )

    assert device == "cpu"
    assert spec == _CPU


def test_native_default_preserves_safe_cuda_device_id() -> None:
    allowed = [{"name": "CUDA", "options": {"device_id": "12"}}]
    verified = copy.deepcopy(allowed)

    device, spec = resolve_native_device(allowed=allowed, verified=verified)

    assert device == "cuda:12"
    assert spec == {"name": "CUDA", "options": {"device_id": 12}}
    assert allowed[0]["options"]["device_id"] == "12"


def test_explicit_native_device_requires_exact_allowed_and_verified_spec() -> None:
    cuda_two = {"name": "CUDA", "options": {"device_id": 2}}

    with pytest.raises(ArtifactRuntimeError, match="not allowed"):
        resolve_native_device(
            allowed=[_CPU, cuda_two],
            verified=[_CPU, cuda_two],
            device="cuda:3",
        )

    with pytest.raises(ArtifactRuntimeError, match="not release-verified"):
        resolve_native_device(
            allowed=[_CPU, _CUDA, cuda_two],
            verified=[_CPU, cuda_two],
            device="gpu",
        )

    device, spec = resolve_native_device(
        allowed=[_CPU, _CUDA, cuda_two],
        verified=[_CPU, cuda_two],
        device="cuda:2",
    )
    assert device == "cuda:2"
    assert spec == cuda_two


def test_native_contract_rejects_empty_or_inconsistent_authority() -> None:
    with pytest.raises(ArtifactRuntimeError, match="non-empty"):
        resolve_native_device(allowed=[], verified=[_CPU])
    with pytest.raises(ArtifactRuntimeError, match="non-empty"):
        resolve_native_device(allowed=[_CPU], verified=[])
    with pytest.raises(ArtifactRuntimeError, match="no exact allowed-and-verified"):
        resolve_native_device(allowed=[_CPU], verified=[_CUDA])
    with pytest.raises(ArtifactRuntimeError, match="exact subset"):
        resolve_native_device(allowed=[_CPU], verified=[_CPU, _CUDA])


@pytest.mark.parametrize(
    "device",
    ["", "cpu:1", "cuda:-1", "cuda:1.0", "gpu:0", "mps", 0, False],
)
def test_native_contract_rejects_unsupported_device_values(device) -> None:  # noqa: ANN001
    with pytest.raises(ArtifactRuntimeError, match="Unsupported native device override"):
        resolve_native_device(allowed=[_CPU], verified=[_CPU], device=device)


@pytest.mark.parametrize(
    "provider",
    [
        {"name": "CPUExecutionProvider", "options": {}},
        {"name": "cpu", "options": {}},
        {"name": "CPU", "options": {"device_id": 0}},
        {"name": "CUDA", "options": {"gpu_mem_limit": 1}},
        {"name": "CUDA", "options": {"device_id": True}},
        {"name": "CUDA", "options": {"device_id": -1}},
        {"name": "CUDA", "options": {"device_id": "-1"}},
        {"name": "CUDA", "options": {"device_id": 1.5}},
        {"name": "CUDA", "options": []},
        {"name": "CPU", "options": {}, "library_path": "/tmp/provider.so"},
        "CPU",
    ],
)
def test_native_contract_rejects_noncanonical_or_unsafe_provider_specs(provider) -> None:
    with pytest.raises(ArtifactRuntimeError):
        resolve_native_device(allowed=[provider], verified=[provider])


def test_native_contract_rejects_duplicate_normalized_provider_specs() -> None:
    with pytest.raises(ArtifactRuntimeError, match="duplicate provider spec"):
        resolve_native_device(
            allowed=[
                {"name": "CUDA", "options": {"device_id": 1}},
                {"name": "CUDA", "options": {"device_id": "1"}},
            ],
            verified=[{"name": "CUDA", "options": {"device_id": 1}}],
        )
