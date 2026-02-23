import json


def _write_png(path, *, value: int = 128) -> None:
    import cv2
    import numpy as np

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_list_dataset_categories_prefers_on_disk_subset(tmp_path):
    from pyimgano.pipelines.run_benchmark import list_dataset_categories

    root = tmp_path / "mvtec"
    (root / "bottle").mkdir(parents=True)
    (root / "not_a_category").mkdir(parents=True)

    cats = list_dataset_categories(dataset="mvtec", root=str(root))
    assert cats == ["bottle"]


def test_list_dataset_categories_visa_uses_visa_pytorch_root(tmp_path):
    from pyimgano.pipelines.run_benchmark import list_dataset_categories

    root = tmp_path / "visa"
    (root / "visa_pytorch" / "candle").mkdir(parents=True)
    (root / "visa_pytorch" / "capsules").mkdir(parents=True)

    cats = list_dataset_categories(dataset="visa", root=str(root))
    assert cats == ["candle", "capsules"]


def test_cli_oneclick_custom_dataset_writes_run_artifacts(tmp_path, capsys):
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
            "--limit-train",
            "1",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    # Artifacts
    assert (out_dir / "report.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "categories" / "custom" / "report.json").exists()
    assert (out_dir / "categories" / "custom" / "per_image.jsonl").exists()

    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "custom"
    assert payload["category"] == "custom"
    assert payload["model"] == "vision_ecod"

    lines = (out_dir / "categories" / "custom" / "per_image.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert {"score", "threshold", "pred", "y_true"}.issubset(rec0.keys())

    stdout = capsys.readouterr().out
    # CLI prints JSON summary (stdout) when --output is not used.
    assert "\"dataset\"" in stdout


def test_cli_oneclick_category_all_writes_aggregated_report(tmp_path):
    from pyimgano.cli import main

    root = tmp_path / "mvtec"
    cat = "bottle"
    _write_png(root / cat / "train" / "good" / "train_0.png", value=120)
    _write_png(root / cat / "test" / "good" / "good_0.png", value=120)
    _write_png(root / cat / "test" / "crack" / "bad_0.png", value=240)

    out_dir = tmp_path / "run_out"
    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(root),
            "--category",
            "all",
            "--model",
            "vision_ecod",
            "--device",
            "cpu",
            "--no-pretrained",
            "--limit-train",
            "1",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "mvtec"
    assert payload["category"] == "all"
    assert payload["categories"] == ["bottle"]
    assert "mean_metrics" in payload
    assert "std_metrics" in payload
    assert "per_category" in payload
    assert (out_dir / "categories" / "bottle" / "per_image.jsonl").exists()
