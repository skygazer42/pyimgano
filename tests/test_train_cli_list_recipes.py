import json


def test_train_cli_list_recipes_outputs_text(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "industrial-adapt" in out


def test_train_cli_list_recipes_outputs_json(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert "industrial-adapt" in parsed


def test_train_cli_recipe_info_outputs_json(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "industrial-adapt"

