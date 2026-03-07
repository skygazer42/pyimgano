import json


def _write_png(path, *, value: int = 128) -> None:
    import cv2
    import numpy as np

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_cli_supports_config_file_dict_and_cli_overrides(tmp_path):
    from pyimgano.cli import main

    # Create a tiny dataset with >2 train images so --limit-train is observable.
    root = tmp_path / "custom_ds"
    for i in range(3):
        _write_png(root / "train" / "normal" / f"train_{i}.png", value=120)
    for i in range(2):
        _write_png(root / "test" / "normal" / f"good_{i}.png", value=120)
        _write_png(root / "test" / "anomaly" / f"bad_{i}.png", value=240)

    cfg = {
        "dataset": "custom",
        "root": str(root),
        "category": "custom",
        "model": "vision_ecod",
        "device": "cpu",
        "pretrained": False,
        "save_run": False,
        "per_image_jsonl": False,
        # Base config would run 2 train items...
        "limit_train": 2,
        "limit_test": 2,
    }
    cfg_path = tmp_path / "bench_cfg.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    out_path = tmp_path / "out.json"
    code = main(
        [
            "--config",
            str(cfg_path),
            # ...but CLI overrides to 1.
            "--limit-train",
            "1",
            "--output",
            str(out_path),
        ]
    )
    assert code == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "custom"
    assert int(payload["dataset_summary"]["train_count"]) == 1


def test_cli_supports_config_file_list(tmp_path):
    from pyimgano.cli import main

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    cfg = [
        "--dataset",
        "custom",
        "--root",
        str(root),
        "--category",
        "custom",
        "--model",
        "vision_ecod",
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-save-run",
        "--no-per-image-jsonl",
        "--limit-train",
        "1",
        "--limit-test",
        "2",
    ]
    cfg_path = tmp_path / "bench_cfg_list.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    out_path = tmp_path / "out.json"
    code = main(["--config", str(cfg_path), "--output", str(out_path)])
    assert code == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "custom"
