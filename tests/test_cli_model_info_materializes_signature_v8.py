import json


def test_cli_model_info_json_includes_real_constructor_kwargs(capsys):
    from pyimgano.cli import main

    code = main(["--model-info", "vision_patchcore", "--json"])
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)

    # Under the v7 lazy registry, placeholders have a generic (*args, **kwargs)
    # signature. `--model-info` should materialize the real constructor so users
    # see an actionable signature.
    assert "device" in payload["accepted_kwargs"]
    assert "device" in payload["signature"]

