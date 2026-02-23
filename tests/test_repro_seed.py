import json


def _write_png(path, *, value: int = 128) -> None:
    import cv2
    import numpy as np

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_cli_seed_persisted_in_config(tmp_path, capsys):
    from pyimgano.cli import main

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    out_dir = tmp_path / "run_out"
    code = main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--model",
            "vision_ecod",
            "--device",
            "cpu",
            "--no-pretrained",
            "--seed",
            "123",
            "--limit-train",
            "1",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    capsys.readouterr()

    config_path = out_dir / "config.json"
    assert config_path.exists()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["config"]["seed"] == 123

