import json


def _write_png(path, *, value: int = 128) -> None:
    import cv2
    import numpy as np

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_run_writes_environment_json(tmp_path):
    from pyimgano.pipelines.run_benchmark import run_benchmark

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    out_dir = tmp_path / "run_out"
    payload = run_benchmark(
        dataset="custom",
        root=str(root),
        category="custom",
        model="vision_ecod",
        device="cpu",
        pretrained=False,
        limit_train=1,
        limit_test=2,
        output_dir=str(out_dir),
        save_run=True,
        per_image_jsonl=False,
    )
    assert payload["dataset"] == "custom"

    env_path = out_dir / "environment.json"
    assert env_path.exists()
    env = json.loads(env_path.read_text(encoding="utf-8"))
    assert {"timestamp_utc", "python", "platform", "packages"}.issubset(env.keys())

