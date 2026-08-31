from __future__ import annotations

import numpy as np
import pytest

from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
from pyimgano.inference.torchscript_runtime import (
    TorchScriptArtifactRuntime,
    resolve_torchscript_device,
)


def _contracts() -> tuple[dict, dict]:
    return (
        {
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [4, 4],
            "resize": {"mode": "stretch", "interpolation": "bilinear"},
            "scale": {"divisor": 255.0},
        },
        {
            "score": {
                "name": "score",
                "output_index": 0,
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            },
            "anomaly_map": {
                "name": "map",
                "output_index": 1,
                "layout": "NHW",
                "resize_to_source": True,
            },
        },
    )


def test_torchscript_artifact_runtime_executes_declared_outputs(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class Detector(torch.nn.Module):
        def forward(self, value):  # noqa: ANN001, ANN201
            return value.mean(dim=(1, 2, 3)), value.mean(dim=1)

    model_path = tmp_path / "detector.pt"
    traced = torch.jit.trace(Detector().eval(), torch.zeros((1, 3, 4, 4)))
    traced.save(str(model_path))
    input_contract, output_contract = _contracts()
    runtime = TorchScriptArtifactRuntime(
        model_path,
        input_contract=input_contract,
        output_contract=output_contract,
        allowed_providers=[{"name": "CPU", "options": {}}],
        verified_providers=[{"name": "CPU", "options": {}}],
        device="cpu",
        trust_checkpoint=True,
    )
    image = np.full((7, 5, 3), 255, dtype=np.uint8)

    scores, maps = runtime.score_and_maps([image])

    np.testing.assert_allclose(scores, [1.0], atol=1e-6)
    assert maps is not None and maps.shape == (1, 7, 5)
    np.testing.assert_allclose(maps, 1.0, atol=1e-6)
    assert runtime.runtime_info["device"] == "cpu"


class _FakeCuda:
    def __init__(self, *, available: bool = True, count: int = 2) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = True, cuda_count: int = 2) -> None:
        self.cuda = _FakeCuda(available=cuda_available, count=cuda_count)


def test_torchscript_device_defaults_to_first_allowed_verified_intersection() -> None:
    selected, spec = resolve_torchscript_device(
        _FakeTorch(),
        allowed=[
            {"name": "CUDA", "options": {"device_id": 1}},
            {"name": "CPU", "options": {}},
        ],
        verified=[{"name": "CPU", "options": {}}],
    )

    assert selected == "cpu"
    assert spec == {"name": "CPU", "options": {}}


def test_explicit_torchscript_device_requires_exact_allowed_and_verified_spec() -> None:
    with pytest.raises(ArtifactRuntimeError, match="not allowed"):
        resolve_torchscript_device(
            _FakeTorch(),
            allowed=[{"name": "CPU", "options": {}}],
            verified=[{"name": "CPU", "options": {}}],
            device="cuda",
        )

    with pytest.raises(ArtifactRuntimeError, match="not release-verified"):
        resolve_torchscript_device(
            _FakeTorch(),
            allowed=[
                {"name": "CPU", "options": {}},
                {"name": "CUDA", "options": {}},
            ],
            verified=[{"name": "CPU", "options": {}}],
            device="GPU",
        )


def test_torchscript_device_rejects_invalid_authority_or_unavailable_cuda() -> None:
    with pytest.raises(ArtifactRuntimeError, match="exact subset"):
        resolve_torchscript_device(
            _FakeTorch(),
            allowed=[{"name": "CPU", "options": {}}],
            verified=[{"name": "CUDA", "options": {}}],
        )

    with pytest.raises(ArtifactRuntimeError, match="CUDA device is unavailable"):
        resolve_torchscript_device(
            _FakeTorch(cuda_available=False),
            allowed=[{"name": "CUDA", "options": {}}],
            verified=[{"name": "CUDA", "options": {}}],
        )

    with pytest.raises(ArtifactRuntimeError, match="CUDA device 4 is unavailable"):
        resolve_torchscript_device(
            _FakeTorch(cuda_count=1),
            allowed=[{"name": "CUDA", "options": {"device_id": 4}}],
            verified=[{"name": "CUDA", "options": {"device_id": 4}}],
        )


def test_torchscript_runtime_rejects_untrusted_archive_before_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pyimgano.utils.torchscript_safe as safe

    called = False

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal called
        called = True
        raise AssertionError("load_module must not be reached")

    monkeypatch.setattr(safe, "load_module", fail_if_called)
    input_contract, output_contract = _contracts()

    with pytest.raises(ArtifactRuntimeError, match="trust_checkpoint=True"):
        TorchScriptArtifactRuntime(
            tmp_path / "untrusted.pt",
            input_contract=input_contract,
            output_contract=output_contract,
            allowed_providers=[{"name": "CPU", "options": {}}],
            verified_providers=[{"name": "CPU", "options": {}}],
        )

    assert called is False


def test_torchscript_runtime_rejects_non_floating_outputs() -> None:
    torch = pytest.importorskip("torch")

    class IntegerDetector:
        def to(self, _device):  # noqa: ANN001, ANN201
            return self

        def eval(self):  # noqa: ANN201
            return self

        def __call__(self, value):  # noqa: ANN001, ANN201
            count = int(value.shape[0])
            return (
                torch.ones((count,), dtype=torch.int64),
                torch.ones((count, 4, 4), dtype=torch.int64),
            )

    input_contract, output_contract = _contracts()
    runtime = TorchScriptArtifactRuntime(
        "injected.pt",
        input_contract=input_contract,
        output_contract=output_contract,
        allowed_providers=[{"name": "CPU", "options": {}}],
        verified_providers=[{"name": "CPU", "options": {}}],
        torch_module=torch,
        model=IntegerDetector(),
    )

    with pytest.raises(ArtifactRuntimeError, match="must be floating point"):
        runtime.decision_function([np.zeros((4, 4, 3), dtype=np.uint8)])
