import json


def test_pyim_list_model_presets_delegates_to_discovery_service(monkeypatch, capsys):
    from pyimgano.pyim_cli import main
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_names",
        lambda **_kwargs: ["delegated-preset"],
    )

    code = main(["--list", "model-presets"])
    assert code == 0
    assert "delegated-preset" in capsys.readouterr().out


def test_pyim_list_model_presets_supports_family_filter(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "model-presets", "--family", "graph"])
    assert code == 0
    out = capsys.readouterr().out
    assert "industrial-structural-rgraph" in out
    assert "industrial-structural-lof" not in out


def test_pyim_list_model_presets_outputs_json_metadata(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "model-presets", "--family", "neighbors", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(
        item["name"] == "industrial-structural-lof" and "neighbors" in item["tags"]
        for item in payload
    )


def test_pyim_list_model_presets_json_delegates_to_discovery_service(monkeypatch, capsys):
    from pyimgano.pyim_cli import main
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_infos_payload",
        lambda **_kwargs: [{"name": "delegated-preset", "tags": ["graph"]}],
    )

    code = main(["--list", "model-presets", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"name": "delegated-preset", "tags": ["graph"]}]


def test_pyim_model_preset_json_uses_shared_rendering_helper(monkeypatch):
    import pyimgano.pyim_app as pyim_app
    import pyimgano.services.discovery_service as discovery_service
    from pyimgano.pyim_contracts import PyimListPayload

    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_names",
        lambda **_kwargs: ["delegated-preset"],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_infos_payload",
        lambda **_kwargs: [{"name": "delegated-preset", "tags": ["graph"]}],
    )

    calls = []
    monkeypatch.setattr(
        pyim_app,
        "pyim_cli_rendering",
        type(
            "_StubPyimCliRendering",
            (),
            {
                "emit_pyim_list_payload": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 23
                )
            },
        ),
        raising=False,
    )

    code = pyim_app.run_pyim_command(
        pyim_app.PyimCommand(
            list_kind="model-presets",
            json_output=True,
        )
    )
    assert code == 23
    assert len(calls) == 1
    payload, kwargs = calls[0]
    assert isinstance(payload, PyimListPayload)
    assert payload.model_presets == ["delegated-preset"]
    assert payload.model_preset_infos == [{"name": "delegated-preset", "tags": ["graph"]}]
    assert kwargs == {"list_kind": "model-presets", "json_output": True}


def test_pyim_model_preset_listing_delegates_raw_filters_to_discovery_service(
    monkeypatch, capsys
):
    from pyimgano.pyim_cli import main
    import pyimgano.services.discovery_service as discovery_service

    service_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_names",
        lambda **kwargs: service_calls.append(("names", dict(kwargs))) or ["delegated-preset"],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_infos_payload",
        lambda **kwargs: service_calls.append(("infos", dict(kwargs)))
        or [{"name": "delegated-preset", "tags": ["graph"]}],
    )

    code = main(["--list", "model-presets", "--family", "graph", "--tags", "embeddings,gaussian"])
    assert code == 0
    assert "delegated-preset" in capsys.readouterr().out
    assert service_calls == [
        ("names", {"tags": ["embeddings,gaussian"], "family": "graph"}),
        ("infos", {"tags": ["embeddings,gaussian"], "family": "graph"}),
    ]
