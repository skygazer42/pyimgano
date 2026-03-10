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

