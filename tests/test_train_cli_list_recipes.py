import json


def test_train_cli_list_recipes_outputs_text(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "industrial-adapt" in out
    assert "classical-structural-ecod" in out


def test_train_cli_list_recipes_outputs_json(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert "industrial-adapt" in parsed
    assert "classical-structural-ecod" in parsed


def test_train_cli_list_recipes_uses_shared_listing_helper(monkeypatch):
    import pyimgano.train_cli as train_cli

    monkeypatch.setattr(train_cli, "list_recipes", lambda: ["recipe-a", "recipe-b"])

    calls = []
    monkeypatch.setattr(
        train_cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {
                "emit_listing": staticmethod(
                    lambda items, **kwargs: calls.append((list(items), kwargs)) or 73
                )
            },
        ),
        raising=False,
    )

    code = train_cli.main(["--list-recipes", "--json"])
    assert code == 73
    assert calls == [(["recipe-a", "recipe-b"], {"json_output": True})]


def test_train_cli_recipe_info_outputs_json(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "industrial-adapt"
