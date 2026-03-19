import json


def test_train_cli_dry_run_prints_human_summary_without_writing(tmp_path, capsys):
    from pyimgano.train_cli import main

    out_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {
                    "name": "custom",
                    "root": "/tmp/data",
                    "category": "widget",
                    "resize": [16, 16],
                    "input_mode": "paths",
                },
                "model": {"name": "vision_patchcore", "device": "cpu"},
                "training": {
                    "enabled": True,
                    "epochs": 3,
                    "batch_size": 8,
                    "num_workers": 2,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "criterion_name": "mae",
                },
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
    assert "Dry Run Summary" in out
    assert "[RUN]" in out
    assert "[CFG]" in out
    assert "[OPT]" in out
    assert "[OUT]" in out
    assert "engine=dry-run" in out
    assert "recipe=industrial-adapt" in out
    assert "data=custom/widget" in out
    assert "model=vision_patchcore" in out
    assert "imgsz=16x16" in out
    assert "device=cpu" in out
    assert "epochs=3" in out
    assert "batch=8" in out
    assert "workers=2" in out
    assert "optimizer=adamw" in out
    assert "scheduler=cosine" in out
    assert "criterion=mae" in out
    assert "save_run=True" in out
    assert "output_dir=" in out
    assert "input=paths" in out


def test_train_cli_dry_run_json_emits_effective_config_payload(tmp_path, capsys):
    from pyimgano.train_cli import main

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--dry-run", "--json"])
    assert code == 0

    assert json.loads(capsys.readouterr().out)["config"]["recipe"] == "industrial-adapt"


def test_train_cli_dry_run_delegates_to_train_service_in_json_mode(tmp_path, capsys, monkeypatch):
    from pyimgano.train_cli import main
    import pyimgano.services.train_service as train_service

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        train_service,
        "build_train_dry_run_payload",
        lambda _request: {"config": {"recipe": "delegated-train"}},
    )

    code = main(["--config", str(cfg_path), "--dry-run", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["config"]["recipe"] == "delegated-train"
