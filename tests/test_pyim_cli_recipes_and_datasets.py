from __future__ import annotations

import json


def test_pyim_list_recipes_outputs_json_infos(capsys) -> None:
    from pyimgano.pyim_cli import main

    code = main(["--list", "recipes", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(
        item.get("name") == "industrial-adapt"
        and isinstance((item.get("metadata") or {}).get("description", None), str)
        for item in payload
    )


def test_pyim_list_datasets_outputs_json_payload(capsys) -> None:
    from pyimgano.pyim_cli import main

    code = main(["--list", "datasets", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(item.get("name") == "custom" for item in payload)
    assert all(
        isinstance(item.get("name"), str)
        and isinstance(item.get("description"), str)
        and isinstance(item.get("requires_category"), bool)
        for item in payload
    )


def test_pyim_list_all_json_includes_recipes_and_datasets(capsys) -> None:
    from pyimgano.pyim_cli import main

    code = main(["--list", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    assert "recipes" in payload
    assert "datasets" in payload

    recipes = payload["recipes"]
    datasets = payload["datasets"]
    assert isinstance(recipes, list)
    assert isinstance(datasets, list)
    assert any(item.get("name") == "industrial-adapt" for item in recipes)
    assert any(item.get("name") == "custom" for item in datasets)


def test_pyim_list_all_json_delegates_payload_collection_to_pyim_service(
    monkeypatch, capsys
) -> None:
    import pyimgano.pyim_app as pyim_app
    from pyimgano.services.pyim_service import PyimListPayload

    calls = []

    monkeypatch.setattr(
        pyim_app,
        "pyim_service",
        type(
            "_StubPyimService",
            (),
            {
                "collect_pyim_listing_payload": staticmethod(
                    lambda request: calls.append(dict(request.__dict__))
                    or PyimListPayload(
                        models=["delegated-model"],
                        families=[{"name": "neighbors", "model_count": 1, "description": "d"}],
                        recipes=[{"name": "industrial-adapt", "metadata": {}}],
                        datasets=[
                            {
                                "name": "custom",
                                "description": "Custom dataset",
                                "requires_category": False,
                            }
                        ],
                    )
                ),
            },
        ),
        raising=False,
    )

    code = pyim_app.run_pyim_command(
        pyim_app.PyimCommand(
            list_kind="all",
            json_output=True,
        )
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == ["delegated-model"]
    assert payload["recipes"] == [{"name": "industrial-adapt", "metadata": {}}]
    assert payload["datasets"] == [
        {
            "name": "custom",
            "description": "Custom dataset",
            "requires_category": False,
        }
    ]
    assert calls == [
        {
            "list_kind": "all",
            "tags": None,
            "family": None,
            "algorithm_type": None,
            "year": None,
            "deployable_only": False,
            "objective": None,
            "selection_profile": None,
            "topk": None,
            "include_core_sections": True,
            "include_recipes": True,
            "include_datasets": True,
        }
    ]


def test_pyim_list_recipes_delegates_payload_collection_to_pyim_service(
    monkeypatch, capsys
) -> None:
    import pyimgano.pyim_app as pyim_app
    from pyimgano.services.pyim_service import PyimListPayload

    calls = []

    monkeypatch.setattr(
        pyim_app,
        "pyim_service",
        type(
            "_StubPyimService",
            (),
            {
                "collect_pyim_listing_payload": staticmethod(
                    lambda request: calls.append(dict(request.__dict__))
                    or PyimListPayload(recipes=[{"name": "delegated-recipe", "metadata": {}}])
                ),
            },
        ),
        raising=False,
    )

    code = pyim_app.run_pyim_command(
        pyim_app.PyimCommand(
            list_kind="recipes",
            json_output=True,
        )
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"name": "delegated-recipe", "metadata": {}}]
    assert calls == [
        {
            "list_kind": "recipes",
            "tags": None,
            "family": None,
            "algorithm_type": None,
            "year": None,
            "deployable_only": False,
            "objective": None,
            "selection_profile": None,
            "topk": None,
            "include_core_sections": False,
            "include_recipes": True,
            "include_datasets": False,
        }
    ]


def test_pyim_list_datasets_delegates_payload_collection_to_pyim_service(
    monkeypatch, capsys
) -> None:
    import pyimgano.pyim_app as pyim_app
    from pyimgano.services.pyim_service import PyimListPayload

    calls = []

    monkeypatch.setattr(
        pyim_app,
        "pyim_service",
        type(
            "_StubPyimService",
            (),
            {
                "collect_pyim_listing_payload": staticmethod(
                    lambda request: calls.append(dict(request.__dict__))
                    or PyimListPayload(
                        datasets=[
                            {
                                "name": "delegated-dataset",
                                "description": "Delegated dataset",
                                "requires_category": False,
                            }
                        ]
                    )
                ),
            },
        ),
        raising=False,
    )

    code = pyim_app.run_pyim_command(
        pyim_app.PyimCommand(
            list_kind="datasets",
            json_output=True,
        )
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "name": "delegated-dataset",
            "description": "Delegated dataset",
            "requires_category": False,
        }
    ]
    assert calls == [
        {
            "list_kind": "datasets",
            "tags": None,
            "family": None,
            "algorithm_type": None,
            "year": None,
            "deployable_only": False,
            "objective": None,
            "selection_profile": None,
            "topk": None,
            "include_core_sections": False,
            "include_recipes": False,
            "include_datasets": True,
        }
    ]
