import json


def test_train_cli_dry_run_prints_effective_config_without_writing(tmp_path, capsys):
    from pyimgano.train_cli import main

    out_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"output_dir": str(out_dir), "save_run": True},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--dry-run"])
    assert code == 0

    # Dry-run should not create run directories.
    assert out_dir.exists() is False

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "config" in payload
    assert payload["config"]["recipe"] == "industrial-adapt"

