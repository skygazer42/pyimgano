import json


def _write_png(path, *, value: int = 128) -> None:
    import numpy as np
    from PIL import Image

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def _raise_fit_called(*args, **kwargs):  # noqa: ANN002, ANN003 - test helper
    raise RuntimeError("fit() should not be called when using --load-detector")


def test_cli_can_save_detector_pickle(tmp_path) -> None:
    from pyimgano.cli import main
    from pyimgano.serialization import load_detector

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
            "--save-detector",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    detector_path = out_dir / "detector.pkl"
    assert detector_path.exists()

    detector = load_detector(detector_path)
    assert hasattr(detector, "decision_function")

    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["detector_path"] == str(detector_path)


def test_cli_can_load_detector_pickle_and_skip_fit(tmp_path) -> None:
    from pyimgano.cli import main
    from pyimgano.models.registry import create_model
    from pyimgano.serialization import save_detector

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    detector = create_model("vision_ecod", contamination=0.1)
    detector.fit([str(root / "train" / "normal" / "train_0.png")])

    # If the benchmark runner ever calls fit(), raise.
    detector.fit = _raise_fit_called  # type: ignore[method-assign]

    detector_path = tmp_path / "detector.pkl"
    save_detector(detector_path, detector)

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
            "--load-detector",
            str(detector_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["loaded_detector_path"] == str(detector_path)
