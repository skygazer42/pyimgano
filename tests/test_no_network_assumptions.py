from __future__ import annotations


def test_import_and_registry_discovery_do_not_require_network(monkeypatch) -> None:
    """Guardrail: unit tests should not rely on external network access.

    This is intentionally a light guard: it blocks common connection helpers
    and then imports registry discovery surfaces.
    """

    import socket
    import urllib.request

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201 - test helper
        raise AssertionError("Network access is forbidden in unit tests.")

    # Common helpers used by many libs.
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=True)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked, raising=True)

    import pyimgano  # noqa: F401
    import pyimgano.models as models

    # Registry discovery should be local-only.
    names = models.list_models()
    assert isinstance(names, list)
    assert "vision_ecod" in names

