from __future__ import annotations

from pyimgano.models.introspection import model_info
from pyimgano.models.registry import ModelRegistry


def test_model_info_exposes_signature_and_kwargs() -> None:
    def _ctor(*, a: int = 1, b: str = "x") -> object:
        return object()

    registry = ModelRegistry()
    registry.register("toy_model", _ctor, tags=["toy"], metadata={"description": "toy"})

    info = model_info(registry, "toy_model")
    assert info["name"] == "toy_model"
    assert "toy" in info["tags"]
    assert info["metadata"]["description"] == "toy"
    assert "a" in info["accepted_kwargs"]
    assert "b" in info["accepted_kwargs"]
    assert info["accepts_var_kwargs"] is False
    assert "a" in info["signature"]


def test_model_info_handles_var_kwargs_constructors() -> None:
    def _ctor(**kwargs: object) -> object:  # noqa: ARG001 - signature is the behavior
        return object()

    registry = ModelRegistry()
    registry.register("var_kwargs", _ctor)

    info = model_info(registry, "var_kwargs")
    assert info["name"] == "var_kwargs"
    assert info["accepts_var_kwargs"] is True


def test_registry_model_info_method_matches_introspection() -> None:
    def _ctor(*, x: int = 0) -> object:
        return object()

    registry = ModelRegistry()
    registry.register("toy", _ctor, tags=["toy"])

    via_registry = registry.model_info("toy")
    via_introspection = model_info(registry, "toy")
    assert via_registry == via_introspection

