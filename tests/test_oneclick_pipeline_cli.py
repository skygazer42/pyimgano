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


def test_oneclick_pipeline_supports_numpy_input_mode(tmp_path):
    import numpy as np

    from pyimgano.models.registry import MODEL_REGISTRY
    from pyimgano.pipelines.run_benchmark import run_benchmark

    class _DummyNumpyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            return self

        def decision_function(self, X):  # noqa: ANN001
            scores: list[float] = []
            for item in list(X):
                arr = np.asarray(item)
                scores.append(float(arr.mean()))
            return np.asarray(scores, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_dummy_numpy_detector",
        _DummyNumpyDetector,
        tags=("numpy",),
        overwrite=True,
    )

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    payload = run_benchmark(
        dataset="custom",
        root=str(root),
        category="custom",
        model="test_dummy_numpy_detector",
        input_mode="numpy",
        device="cpu",
        pretrained=False,
        save_run=False,
        per_image_jsonl=False,
        limit_train=1,
        limit_test=2,
    )
    assert payload["dataset"] == "custom"
    assert payload["category"] == "custom"
    assert payload["model"] == "test_dummy_numpy_detector"
    assert payload["input_mode"] == "numpy"
    assert "results" in payload


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
    assert payload["schema_version"] == 1
    assert "timestamp_utc" in payload
    assert "pyimgano_version" in payload

    lines = (out_dir / "categories" / "custom" / "per_image.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert {"score", "threshold", "pred", "y_true"}.issubset(rec0.keys())

    stdout = capsys.readouterr().out
    # CLI prints JSON summary (stdout) when --output is not used.
    assert "\"dataset\"" in stdout


def test_cli_oneclick_custom_dataset_validates_structure(tmp_path, capsys):
    from pyimgano.cli import main

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    # Missing: test/anomaly/

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
    assert code != 0

    err = capsys.readouterr().err.lower()
    assert "invalid custom dataset structure" in err


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
    assert payload["schema_version"] == 1
    assert "timestamp_utc" in payload
    assert "pyimgano_version" in payload
    assert "mean_metrics" in payload
    assert "std_metrics" in payload
    assert "per_category" in payload
    assert (out_dir / "categories" / "bottle" / "per_image.jsonl").exists()
