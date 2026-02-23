from __future__ import annotations

from pyimgano.models.capabilities import compute_model_capabilities
from pyimgano.models.registry import ModelRegistry


def test_capabilities_include_numpy_input_mode_when_tagged() -> None:
    class _NumpyModel:
        def __init__(self, *, k: int = 1) -> None:
            self.k = int(k)

    registry = ModelRegistry()
    registry.register("numpy_model", _NumpyModel, tags=["numpy"])

    caps = compute_model_capabilities(registry.info("numpy_model"))
    assert "paths" in caps.input_modes
    assert "numpy" in caps.input_modes


def test_capabilities_detect_pixel_map_via_method_presence() -> None:
    class _PixelModel:
        def predict_anomaly_map(self, X):  # noqa: ANN001 - test stub
            raise NotImplementedError

    registry = ModelRegistry()
    registry.register("pixel", _PixelModel)

    caps = compute_model_capabilities(registry.info("pixel"))
    assert caps.supports_pixel_map is True


def test_capabilities_detect_checkpoint_support_from_signature() -> None:
    def _ctor(*, checkpoint_path: str | None = None) -> object:  # noqa: ARG001
        return object()

    registry = ModelRegistry()
    registry.register("checkpointable", _ctor)

    caps = compute_model_capabilities(registry.info("checkpointable"))
    assert caps.requires_checkpoint is False
    assert caps.supports_checkpoint is True


def test_capabilities_respect_requires_checkpoint_metadata() -> None:
    class _Model:
        pass

    registry = ModelRegistry()
    registry.register("requires", _Model, metadata={"requires_checkpoint": True})

    caps = compute_model_capabilities(registry.info("requires"))
    assert caps.requires_checkpoint is True
    assert caps.supports_checkpoint is True


def test_capabilities_mark_classical_models_as_save_load_capable() -> None:
    def _ctor() -> object:
        return object()

    registry = ModelRegistry()
    registry.register("classical", _ctor, tags=["vision", "classical"])

    caps = compute_model_capabilities(registry.info("classical"))
    assert caps.supports_save_load is True

