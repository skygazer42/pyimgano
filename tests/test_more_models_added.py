def test_cli_list_models_includes_added_pyod_models(capsys):
    from pyimgano.cli import main

    code = main(["--list-models"])
    assert code == 0
    out = capsys.readouterr().out

    # New model wrappers added in this change.
    assert "vision_cd" in out
    assert "vision_dif" in out
    assert "vision_lunar" in out
    assert "vision_rgraph" in out
    assert "vision_sampling" in out
    assert "vision_so_gaal" in out
    assert "vision_so_gaal_new" in out

