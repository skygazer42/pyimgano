from __future__ import annotations

import importlib.metadata as md


def test_load_plugins_entrypoints_registers_models(monkeypatch) -> None:
    # Plugins should be opt-in: importing pyimgano shouldn't auto-load them.
    import pyimgano.models as models

    assert "plugin_dummy_model" not in set(models.list_models())

    # Patch entry_points() to expose our dummy plugin entrypoint.
    ep = md.EntryPoint(
        name="dummy",
        value="tests.dummy_pyimgano_plugin:register",
        group="pyimgano.plugins",
    )

    def _fake_entry_points():  # noqa: ANN001, ANN201 - stdlib compat shim
        class _EPS(list):
            def select(self, *, group: str):  # noqa: ANN001, ANN201
                if str(group) == "pyimgano.plugins":
                    return [ep]
                return []

        return _EPS([ep])

    monkeypatch.setattr(md, "entry_points", _fake_entry_points, raising=True)

    # This will fail until `pyimgano.plugins.load_plugins()` exists.
    from pyimgano.plugins import load_plugins

    results = load_plugins(groups=("pyimgano.plugins",))
    assert any(r.get("name") == "dummy" and r.get("status") == "loaded" for r in results)

    assert "plugin_dummy_model" in set(models.list_models())
