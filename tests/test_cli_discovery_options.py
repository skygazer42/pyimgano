from __future__ import annotations

import pytest


def test_validate_mutually_exclusive_flags_raises_formatted_error() -> None:
    from pyimgano.cli_discovery_options import validate_mutually_exclusive_flags

    with pytest.raises(
        ValueError,
        match=(
            r"--list-models, --model-info, and --list-model-presets are mutually exclusive\."
        ),
    ):
        validate_mutually_exclusive_flags(
            [
                ("--list-models", True),
                ("--model-info", False),
                ("--list-model-presets", True),
            ]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"algorithm_type": "deep-vision"}, "--type is supported only with --list-models."),
        ({"year": "2024"}, "--year is supported only with --list-models."),
        ({"family": "graph"}, "--family is supported only with --list-models."),
    ],
)
def test_resolve_model_list_discovery_options_rejects_non_list_model_filters(
    kwargs, message: str
) -> None:
    from pyimgano.cli_discovery_options import resolve_model_list_discovery_options

    with pytest.raises(ValueError, match=message):
        resolve_model_list_discovery_options(list_models=False, tags=None, **kwargs)


def test_resolve_model_list_discovery_options_validates_filters(monkeypatch) -> None:
    import pyimgano.cli_discovery_options as cli_discovery_options

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli_discovery_options,
        "resolve_family_tags",
        lambda value: calls.append(("family", value)) or [value],
    )
    monkeypatch.setattr(
        cli_discovery_options,
        "resolve_type_tags",
        lambda value: calls.append(("type", value)) or [value],
    )
    monkeypatch.setattr(
        cli_discovery_options,
        "resolve_year_filter",
        lambda value: calls.append(("year", value)) or value,
    )

    options = cli_discovery_options.resolve_model_list_discovery_options(
        list_models=True,
        tags=["vision", "deep"],
        family="one-to-normal",
        algorithm_type="deep-vision",
        year=2025,
    )

    assert options.tags == ["vision", "deep"]
    assert options.family == "one-to-normal"
    assert options.algorithm_type == "deep-vision"
    assert options.year == "2025"
    assert calls == [
        ("family", "one-to-normal"),
        ("type", "deep-vision"),
        ("year", "2025"),
    ]


def test_resolve_model_list_discovery_options_can_allow_family_without_list_models(
    monkeypatch,
) -> None:
    import pyimgano.cli_discovery_options as cli_discovery_options

    calls: list[str] = []
    monkeypatch.setattr(
        cli_discovery_options,
        "resolve_family_tags",
        lambda value: calls.append(value) or [value],
    )

    options = cli_discovery_options.resolve_model_list_discovery_options(
        list_models=False,
        tags=None,
        family="graph",
        allow_family_without_list_models=True,
    )

    assert options.family == "graph"
    assert calls == ["graph"]
