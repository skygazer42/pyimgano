import json
from types import SimpleNamespace


def test_cli_list_models_outputs_text(capsys):
    from pyimgano.cli import main

    code = main(["--list-models"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out


def test_cli_list_models_outputs_json(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert "vision_patchcore" in parsed


def test_cli_list_models_supports_tags_filter(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--tags", "vision,deep"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out
    assert "vision_abod" not in out


def test_cli_list_models_supports_numpy_capability_tag(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--tags", "numpy"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_spade" in out
    assert "vision_patchcore" in out
    assert "vision_padim" in out
    assert "vision_anomalydino" in out
    assert "vision_softpatch" in out


def test_cli_list_models_supports_core_tag(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--tags", "core"])
    assert code == 0
    out = capsys.readouterr().out
    assert "core_knn" in out
    assert "vision_knn" not in out


def test_cli_list_models_supports_year_and_type_filters(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--year", "2001", "--type", "one-class-svm"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_ocsvm" in out
    assert "vision_lof" not in out


def test_cli_list_models_supports_new_parallel_family_filters(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--year", "2025", "--family", "one-to-normal"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_one_to_normal" in out
    assert "vision_anogen_adapter" not in out


def test_cli_model_info_outputs_text(capsys):
    from pyimgano.cli import main

    code = main(["--model-info", "vision_patchcore"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out
    assert "Signature" in out


def test_cli_model_info_outputs_json(capsys):
    from pyimgano.cli import main

    code = main(["--model-info", "vision_patchcore", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "vision_patchcore"
    assert "signature" in parsed


def test_cli_model_info_uses_shared_discovery_renderer(monkeypatch):
    import pyimgano.cli as cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "build_model_info_payload",
        lambda _name: {
            "name": "delegated-model",
            "tags": [],
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
                    lambda payload, *, json_output: calls.append((payload, json_output)) or 41
                )
            },
        ),
        raising=False,
    )

    code = cli.main(["--model-info", "vision_patchcore"])
    assert code == 41
    assert calls == [
        (
            {
                "name": "delegated-model",
                "tags": [],
                "metadata": {},
                "signature": "()",
                "accepts_var_kwargs": False,
                "accepted_kwargs": [],
            },
            False,
        )
    ]


def test_cli_discovery_flags_are_mutually_exclusive(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--model-info", "vision_patchcore"])
    assert code != 0
    err = capsys.readouterr().err
    assert "mutually" in err.lower() or "exclusive" in err.lower()


def test_cli_list_models_delegates_to_discovery_service(monkeypatch, capsys):
    from pyimgano.cli import main
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **kwargs: ["delegated_model"],
    )

    code = main(["--list-models"])
    assert code == 0
    out = capsys.readouterr().out
    assert "delegated_model" in out


def test_cli_list_models_uses_shared_listing_helper(monkeypatch) -> None:
    import pyimgano.cli as cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **_kwargs: ["delegated-model"],
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
                    lambda items, **kwargs: calls.append((list(items), kwargs)) or 59
                )
            },
        ),
        raising=False,
    )

    code = cli.main(["--list-models", "--json"])
    assert code == 59
    assert calls == [
        (
            ["delegated-model"],
            {"json_output": True, "sort_keys": False},
        )
    ]


def test_cli_list_models_uses_shared_discovery_option_helper(monkeypatch) -> None:
    import pyimgano.cli as cli
    import pyimgano.services.discovery_service as discovery_service

    helper_calls = []

    monkeypatch.setattr(
        cli,
        "cli_discovery_options",
        type(
            "_StubDiscoveryOptions",
            (),
            {
                "validate_mutually_exclusive_flags": staticmethod(
                    lambda flags: helper_calls.append(("flags", list(flags)))
                ),
                "resolve_model_list_discovery_options": staticmethod(
                    lambda **kwargs: helper_calls.append(("resolve", dict(kwargs)))
                    or SimpleNamespace(
                        tags=["normalized-tag"],
                        family="normalized-family",
                        algorithm_type="normalized-type",
                        year="2030",
                    )
                ),
            },
        ),
        raising=False,
    )

    discovery_calls = []
    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **kwargs: discovery_calls.append(dict(kwargs)) or ["delegated-model"],
    )
    monkeypatch.setattr(
        cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {"emit_listing": staticmethod(lambda items, **kwargs: 79)},
        ),
        raising=False,
    )

    code = cli.main(["--list-models"])
    assert code == 79
    assert helper_calls[0][0] == "flags"
    assert helper_calls[1] == (
        "resolve",
        {
            "list_models": True,
            "tags": None,
            "family": None,
            "algorithm_type": None,
            "year": None,
            "allow_family_without_list_models": False,
        },
    )
    assert discovery_calls == [
        {
            "tags": ["normalized-tag"],
            "family": "normalized-family",
            "algorithm_type": "normalized-type",
            "year": "2030",
        }
    ]
