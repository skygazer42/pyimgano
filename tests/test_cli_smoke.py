from pyimgano.cli import main


def test_cli_importable():
    assert callable(main)


def test_cli_pixel_mode_uses_pipeline_alignment(tmp_path, capsys):
    import cv2
    import numpy as np

    # Create a tiny MVTec-style dataset on disk where:
    # - images are 32x32
    # - CLI will load masks resized to 256x256
    # This used to break when CLI stacked detector maps without resizing.
    root = tmp_path / "mvtec"
    cat = "bottle"

    (root / cat / "train" / "good").mkdir(parents=True)
    (root / cat / "test" / "good").mkdir(parents=True)
    (root / cat / "test" / "crack").mkdir(parents=True)
    (root / cat / "ground_truth" / "crack").mkdir(parents=True)

    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(root / cat / "train" / "good" / "train_0.png"), img)
    cv2.imwrite(str(root / cat / "test" / "good" / "good_0.png"), img)

    bad = img.copy()
    bad[8:24, 8:24] = 255
    cv2.imwrite(str(root / cat / "test" / "crack" / "bad_0.png"), bad)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / cat / "ground_truth" / "crack" / "bad_0_mask.png"), mask)

    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(root),
            "--category",
            cat,
            "--model",
            "vision_patchcore",
            "--device",
            "cpu",
            "--no-pretrained",
            "--pixel",
            "--pixel-postprocess",
            "--pixel-post-norm",
            "percentile",
            "--pixel-post-percentiles",
            "1",
            "99",
            "--pixel-post-gaussian-sigma",
            "1.0",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "pixel_metrics" in out


def test_cli_supports_numpy_input_mode_flag(tmp_path, capsys):
    import cv2
    import numpy as np

    from pyimgano.cli import main
    from pyimgano.models.registry import MODEL_REGISTRY

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
        "test_cli_numpy_detector",
        _DummyNumpyDetector,
        tags=("numpy",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    code = main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--model",
            "test_cli_numpy_detector",
            "--input-mode",
            "numpy",
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
    out = capsys.readouterr().out
    assert "\"input_mode\"" in out
