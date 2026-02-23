import json


def test_train_cli_recipe_info_json_includes_callable_path(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "industrial-adapt"
    assert parsed["callable"].endswith(
        "pyimgano.recipes.builtin.industrial_adapt.industrial_adapt"
    )

