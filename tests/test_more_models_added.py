def test_cli_list_models_includes_native_models_added_in_pyod_removal(capsys):
    from pyimgano.cli import main

    code = main(["--list-models"])
    assert code == 0
    out = capsys.readouterr().out
    names = {
        line.strip()
        for line in out.splitlines()
        if line.strip() and line.strip().replace("_", "").isalnum()
    }

    # Native classical detector expansions (ported off PyOD).
    assert "vision_feature_bagging" in names
    assert "vision_lscp" in names
    assert "vision_suod" in names
    assert "vision_rgraph" in names
    assert "vision_sampling" in names

    # PyOD-only wrappers should not be auto-registered in the default registry.
    assert "vision_cd" not in names
    assert "vision_auto_encoder" not in names
    assert "vision_anogan" not in names
    assert "vision_dif" not in names
    assert "vision_lunar" not in names
    assert "vision_so_gaal" not in names
    assert "vision_so_gaal_new" not in names
    assert "vision_mo_gaal" not in names
    assert "vision_xgbod" not in names
