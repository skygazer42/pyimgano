from __future__ import annotations


def test_cli_plugins_flag_triggers_plugin_loading(monkeypatch) -> None:
    import pyimgano.plugins as plugins
    from pyimgano.cli import main

    called = {"n": 0}

    def _stub_load_plugins(
        *, groups=("pyimgano.plugins",), on_error="warn"
    ):  # noqa: ANN001, ANN201
        del groups, on_error
        called["n"] += 1
        return []

    monkeypatch.setattr(plugins, "load_plugins", _stub_load_plugins, raising=True)

    rc = main(["--list-models", "--plugins"])
    assert rc == 0
    assert called["n"] == 1
