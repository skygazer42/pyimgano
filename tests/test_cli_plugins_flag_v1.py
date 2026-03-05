from __future__ import annotations


def test_cli_plugins_flag_triggers_plugin_loading(monkeypatch) -> None:
    import pyimgano.plugins as plugins
    from pyimgano.cli import main

    called = {"n": 0}

    def _stub_load_plugins(
        *, groups=("pyimgano.plugins",), on_error="warn"
    ):  # noqa: ANN001, ANN201
        called["n"] += 1
        return []

    monkeypatch.setattr(plugins, "load_plugins", _stub_load_plugins, raising=True)

    try:
        rc = main(["--list-models", "--plugins"])
    except SystemExit as exc:  # pragma: no cover - current behavior before flag exists
        raise AssertionError("--plugins flag should be accepted by CLI") from exc

    assert rc == 0
    assert called["n"] == 1
