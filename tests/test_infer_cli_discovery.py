from __future__ import annotations

from argparse import Namespace


def test_maybe_run_infer_discovery_command_lists_models(monkeypatch) -> None:
    import pyimgano.infer_cli_discovery as discovery_cli

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "validate_mutually_exclusive_flags",
        lambda flags: calls.append(("validate", flags)),
    )
    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "resolve_model_list_discovery_options",
        lambda **kwargs: type(
            "_Opts",
            (),
            {
                "tags": ["pixel_map"],
                "family": "patchcore",
                "algorithm_type": "vision",
                "year": 2024,
            },
        )(),
    )
    monkeypatch.setattr(
        discovery_cli.discovery_service,
        "list_discovery_model_names",
        lambda **kwargs: calls.append(("models", kwargs)) or ["m1", "m2"],
    )
    monkeypatch.setattr(
        discovery_cli.cli_listing,
        "emit_listing",
        lambda payload, **kwargs: calls.append(("emit", (payload, kwargs))) or 17,
    )

    rc = discovery_cli.maybe_run_infer_discovery_command(
        Namespace(
            list_models=True,
            model_info=None,
            list_model_presets=False,
            model_preset_info=None,
            tags=["pixel_map"],
            family="patchcore",
            algorithm_type="vision",
            year=2024,
            json=False,
        )
    )

    assert rc == 17
    assert calls[-2:] == [
        (
            "models",
            {
                "tags": ["pixel_map"],
                "family": "patchcore",
                "algorithm_type": "vision",
                "year": 2024,
            },
        ),
        ("emit", (["m1", "m2"], {"json_output": False, "sort_keys": False})),
    ]


def test_maybe_run_infer_discovery_command_emits_model_info(monkeypatch) -> None:
    import pyimgano.infer_cli_discovery as discovery_cli

    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "validate_mutually_exclusive_flags",
        lambda flags: None,
    )
    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "resolve_model_list_discovery_options",
        lambda **kwargs: type(
            "_Opts",
            (),
            {"tags": None, "family": None, "algorithm_type": None, "year": None},
        )(),
    )
    monkeypatch.setattr(
        discovery_cli.discovery_service,
        "build_model_info_payload",
        lambda model_name: {"name": model_name, "kind": "model"},
    )
    monkeypatch.setattr(
        discovery_cli.cli_discovery_rendering,
        "emit_signature_payload",
        lambda payload, *, json_output: (payload, json_output),
    )

    rc = discovery_cli.maybe_run_infer_discovery_command(
        Namespace(
            list_models=False,
            model_info="vision_patchcore",
            list_model_presets=False,
            model_preset_info=None,
            tags=None,
            family=None,
            algorithm_type=None,
            year=None,
            json=True,
        )
    )

    assert rc == ({"name": "vision_patchcore", "kind": "model"}, True)


def test_maybe_run_infer_discovery_command_lists_model_presets_json(monkeypatch) -> None:
    import pyimgano.infer_cli_discovery as discovery_cli

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "validate_mutually_exclusive_flags",
        lambda flags: None,
    )
    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "resolve_model_list_discovery_options",
        lambda **kwargs: type(
            "_Opts",
            (),
            {"tags": ["graph"], "family": "distillation", "algorithm_type": None, "year": None},
        )(),
    )
    monkeypatch.setattr(
        discovery_cli.discovery_service,
        "list_model_preset_names",
        lambda **kwargs: calls.append(("names", kwargs)) or ["preset-a", "preset-b"],
    )
    monkeypatch.setattr(
        discovery_cli.discovery_service,
        "list_model_preset_infos_payload",
        lambda **kwargs: calls.append(("infos", kwargs)) or [{"name": "preset-a"}],
    )
    monkeypatch.setattr(
        discovery_cli.cli_listing,
        "emit_listing",
        lambda payload, **kwargs: calls.append(("emit", (payload, kwargs))) or 23,
    )

    rc = discovery_cli.maybe_run_infer_discovery_command(
        Namespace(
            list_models=False,
            model_info=None,
            list_model_presets=True,
            model_preset_info=None,
            tags=["graph"],
            family="distillation",
            algorithm_type=None,
            year=None,
            json=True,
        )
    )

    assert rc == 23
    assert calls == [
        ("names", {"tags": ["graph"], "family": "distillation"}),
        ("infos", {"tags": ["graph"], "family": "distillation"}),
        (
            "emit",
            (
                ["preset-a", "preset-b"],
                {
                    "json_output": True,
                    "json_payload": [{"name": "preset-a"}],
                    "sort_keys": False,
                },
            ),
        ),
    ]


def test_maybe_run_infer_discovery_command_returns_none_for_inference_mode(monkeypatch) -> None:
    import pyimgano.infer_cli_discovery as discovery_cli

    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "validate_mutually_exclusive_flags",
        lambda flags: None,
    )
    monkeypatch.setattr(
        discovery_cli.cli_discovery_options,
        "resolve_model_list_discovery_options",
        lambda **kwargs: type(
            "_Opts",
            (),
            {"tags": None, "family": None, "algorithm_type": None, "year": None},
        )(),
    )

    rc = discovery_cli.maybe_run_infer_discovery_command(
        Namespace(
            list_models=False,
            model_info=None,
            list_model_presets=False,
            model_preset_info=None,
            tags=None,
            family=None,
            algorithm_type=None,
            year=None,
            json=False,
        )
    )

    assert rc is None
