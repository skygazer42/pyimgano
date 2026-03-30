from __future__ import annotations

import pytest


def test_resolve_pyim_list_options_normalizes_filters_and_section_flags(monkeypatch) -> None:
    import pyimgano.pyim_cli_options as pyim_cli_options

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pyim_cli_options,
        "resolve_family_tags",
        lambda value: calls.append(("family", value)) or [value],
    )
    monkeypatch.setattr(
        pyim_cli_options,
        "resolve_type_tags",
        lambda value: calls.append(("type", value)) or [value],
    )
    monkeypatch.setattr(
        pyim_cli_options,
        "resolve_year_filter",
        lambda value: calls.append(("year", value)) or value,
    )

    options = pyim_cli_options.resolve_pyim_list_options(
        list_kind="models",
        tags=["vision", "deep"],
        family="graph",
        algorithm_type="deep-vision",
        year=2025,
        deployable_only=False,
        objective="latency",
        selection_profile="cpu-screening",
        topk=3,
    )

    assert options.list_kind == "models"
    assert options.tags == ["vision", "deep"]
    assert options.family == "graph"
    assert options.algorithm_type == "deep-vision"
    assert options.year == "2025"
    assert options.deployable_only is False
    assert options.objective == "latency"
    assert options.selection_profile == "cpu-screening"
    assert options.topk == 3
    assert options.include_core_sections is True
    assert options.include_recipes is False
    assert options.include_datasets is False
    assert calls == [
        ("family", "graph"),
        ("type", "deep-vision"),
        ("year", "2025"),
    ]


def test_resolve_pyim_list_options_marks_all_sections_for_all_listing() -> None:
    from pyimgano.pyim_cli_options import resolve_pyim_list_options

    options = resolve_pyim_list_options(
        list_kind="all",
        tags=None,
        deployable_only=True,
    )

    assert options.list_kind == "all"
    assert options.include_core_sections is True
    assert options.include_recipes is True
    assert options.include_datasets is True
    assert options.deployable_only is True


def test_resolve_pyim_list_options_marks_recipe_listing_without_core_sections() -> None:
    from pyimgano.pyim_cli_options import resolve_pyim_list_options

    options = resolve_pyim_list_options(
        list_kind="recipes",
        tags=None,
    )

    assert options.list_kind == "recipes"
    assert options.include_core_sections is False
    assert options.include_recipes is True
    assert options.include_datasets is False


def test_pyim_list_options_can_build_neutral_request(monkeypatch) -> None:
    import pyimgano.pyim_cli_options as pyim_cli_options

    calls = []

    class _Request:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        pyim_cli_options,
        "pyim_contracts",
        type(
            "_StubPyimContracts",
            (),
            {"PyimListRequest": _Request},
        ),
        raising=False,
    )

    options = pyim_cli_options.PyimListOptions(
        list_kind="models",
        tags=["vision"],
        family="graph",
        algorithm_type="deep-vision",
        year="2025",
        deployable_only=True,
        objective="balanced",
        selection_profile="benchmark-parity",
        topk=4,
    )

    request = options.to_request()

    assert isinstance(request, _Request)
    assert calls == [
        {
            "list_kind": "models",
            "tags": ["vision"],
            "family": "graph",
            "algorithm_type": "deep-vision",
            "year": "2025",
            "deployable_only": True,
            "objective": "balanced",
            "selection_profile": "benchmark-parity",
            "topk": 4,
        }
    ]


def test_pyim_cli_options_reexports_shared_list_kind_choices() -> None:
    from pyimgano.pyim_cli_options import PYIM_LIST_KIND_CHOICES
    from pyimgano.pyim_list_spec import PYIM_LIST_KIND_CHOICES as SPEC_CHOICES

    assert PYIM_LIST_KIND_CHOICES == SPEC_CHOICES


def test_resolve_pyim_list_options_supports_selection_flags_only_for_model_listings() -> None:
    from pyimgano.pyim_cli_options import resolve_pyim_list_options

    options = resolve_pyim_list_options(
        list_kind="models",
        tags=None,
        objective="localization",
        selection_profile="deploy-readiness",
        topk=2,
    )

    assert options.objective == "localization"
    assert options.selection_profile == "deploy-readiness"
    assert options.topk == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"list_kind": "recipes", "family": "graph"},
            "--family is supported only with --list models, --list model-presets, or --list.",
        ),
        (
            {"list_kind": "preprocessing", "algorithm_type": "deep-vision"},
            "--type is supported only with --list models.",
        ),
        (
            {"list_kind": "all", "year": "2025"},
            "--year is supported only with --list models.",
        ),
        (
            {"list_kind": "models", "deployable_only": True},
            "--deployable-only is supported only with --list preprocessing or --list.",
        ),
        (
            {"list_kind": "families", "objective": "latency"},
            "--objective is supported only with --list models.",
        ),
        (
            {"list_kind": "types", "selection_profile": "cpu-screening"},
            "--selection-profile is supported only with --list models.",
        ),
        (
            {"list_kind": "all", "topk": 2},
            "--topk is supported only with --list models.",
        ),
    ],
)
def test_resolve_pyim_list_options_rejects_invalid_filter_combinations(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from pyimgano.pyim_cli_options import resolve_pyim_list_options

    with pytest.raises(ValueError, match=message):
        resolve_pyim_list_options(tags=None, **kwargs)
