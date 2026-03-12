import json


def test_cli_list_feature_extractors_outputs_text(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--list-feature-extractors"])
    assert code == 0
    out = capsys.readouterr().out
    assert "identity" in out
    assert "torchvision_backbone_gem" in out


def test_cli_list_feature_extractors_outputs_json(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--list-feature-extractors", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert "identity" in parsed
    assert "torchvision_backbone_gem" in parsed


def test_cli_feature_info_outputs_text(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--feature-info", "identity"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Name:" in out
    assert "identity" in out


def test_cli_feature_info_uses_shared_discovery_renderer(monkeypatch) -> None:
    import pyimgano.cli as cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "build_feature_info_payload",
        lambda _name: {
            "name": "identity",
            "tags": ["feature"],
            "metadata": {},
            "signature": "()",
            "accepts_var_kwargs": False,
            "accepted_kwargs": [],
        },
    )

    calls = []
    monkeypatch.setattr(
        cli,
        "cli_discovery_rendering",
        type(
            "_StubDiscoveryRendering",
            (),
            {
                "emit_signature_payload": staticmethod(
                    lambda payload, *, json_output: calls.append((payload, json_output)) or 43
                )
            },
        ),
        raising=False,
    )

    code = cli.main(["--feature-info", "identity"])
    assert code == 43
    assert calls == [
        (
            {
                "name": "identity",
                "tags": ["feature"],
                "metadata": {},
                "signature": "()",
                "accepts_var_kwargs": False,
                "accepted_kwargs": [],
            },
            False,
        )
    ]


def test_cli_feature_discovery_flags_are_mutually_exclusive(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--list-models", "--list-feature-extractors"])
    assert code != 0
    err = capsys.readouterr().err
    assert "mutually" in err.lower() or "exclusive" in err.lower()


def test_cli_list_feature_extractors_uses_shared_listing_helper(monkeypatch) -> None:
    import pyimgano.cli as cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_feature_names",
        lambda **_kwargs: ["feature-a", "feature-b"],
    )

    calls = []
    monkeypatch.setattr(
        cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {
                "emit_listing": staticmethod(
                    lambda items, **kwargs: calls.append((list(items), kwargs)) or 61
                )
            },
        ),
        raising=False,
    )

    code = cli.main(["--list-feature-extractors"])
    assert code == 61
    assert calls == [
        (
            ["feature-a", "feature-b"],
            {"json_output": False, "sort_keys": False},
        )
    ]
