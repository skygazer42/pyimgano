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
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "pixel_metrics" in out
