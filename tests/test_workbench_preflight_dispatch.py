from __future__ import annotations

from types import SimpleNamespace


def test_resolve_preflight_dataset_dispatch_uses_manifest_branch() -> None:
    from pyimgano.workbench.preflight_dispatch import resolve_preflight_dataset_dispatch

    config = SimpleNamespace(dataset=SimpleNamespace(name="manifest"))

    assert resolve_preflight_dataset_dispatch(config=config) == "manifest"


def test_resolve_preflight_dataset_dispatch_uses_non_manifest_branch() -> None:
    from pyimgano.workbench.preflight_dispatch import resolve_preflight_dataset_dispatch

    config = SimpleNamespace(dataset=SimpleNamespace(name="custom"))

    assert resolve_preflight_dataset_dispatch(config=config) == "non_manifest"
