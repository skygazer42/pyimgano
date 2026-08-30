from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.run_golden_mvtec_benchmark import main


def _write_png(path: Path, value: int, *, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask:
        array = np.zeros((16, 16), dtype=np.uint8)
        array[4:12, 4:12] = int(value)
        Image.fromarray(array).save(path)
    else:
        array = np.full((16, 16, 3), int(value), dtype=np.uint8)
        Image.fromarray(array).save(path)


def _make_category(root: Path, category: str) -> Path:
    _write_png(root / category / "train" / "good" / "000.png", 80)
    _write_png(root / category / "test" / "good" / "001.png", 82)
    _write_png(root / category / "test" / "defect" / "002.png", 220)
    mask = root / category / "ground_truth" / "defect" / "002_mask.png"
    _write_png(mask, 255, mask=True)
    return mask


def test_golden_mvtec_check_only_records_content_fingerprints(tmp_path) -> None:
    dataset_root = tmp_path / "mvtec"
    bottle_mask = _make_category(dataset_root, "bottle")
    _make_category(dataset_root, "carpet")

    output_a = tmp_path / "out-a"
    assert (
        main(
            [
                "--root",
                str(dataset_root),
                "--output-dir",
                str(output_a),
                "--check-only",
            ]
        )
        == 0
    )
    manifest_a = json.loads((output_a / "golden_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["dataset"] == "MVTec AD"
    assert manifest_a["categories"] == ["bottle", "carpet"]
    assert manifest_a["dataset_fingerprints"]["bottle"]["file_count"] == 4

    _write_png(bottle_mask, 128, mask=True)
    output_b = tmp_path / "out-b"
    assert (
        main(
            [
                "--root",
                str(dataset_root),
                "--output-dir",
                str(output_b),
                "--check-only",
            ]
        )
        == 0
    )
    manifest_b = json.loads((output_b / "golden_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest_a["dataset_fingerprints"]["bottle"]["sha256"]
        != manifest_b["dataset_fingerprints"]["bottle"]["sha256"]
    )
