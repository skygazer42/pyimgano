from __future__ import annotations

import json

import numpy as np

from pyimgano.robust_cli import main


def test_robust_cli_smoke(tmp_path, capsys) -> None:
    import cv2

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
            "--no-pixel-segf1",
            "--corruptions",
            "lighting",
            "--severities",
            "1",
            "--limit-train",
            "1",
            "--limit-test",
            "2",
        ]
    )
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["dataset"] == "mvtec"
    assert payload["category"] == cat
    assert payload["model"] == "vision_patchcore"

    report = payload["robustness"]
    assert "clean" in report
    assert "corruptions" in report
    assert "lighting" in report["corruptions"]
    assert "severity_1" in report["corruptions"]["lighting"]

