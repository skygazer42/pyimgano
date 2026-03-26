from __future__ import annotations

from types import ModuleType

from tests import conftest as test_conftest


def test_torch_cuda_available_returns_false_for_partial_torch_module(monkeypatch) -> None:
    partial_torch = ModuleType("torch")
    monkeypatch.setitem(__import__("sys").modules, "torch", partial_torch)

    assert test_conftest._torch_cuda_available() is False


def test_seed_torch_if_available_ignores_partial_torch_module(monkeypatch) -> None:
    partial_torch = ModuleType("torch")
    monkeypatch.setitem(__import__("sys").modules, "torch", partial_torch)

    test_conftest._seed_torch_if_available(42)
