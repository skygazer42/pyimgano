import json
from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int = 128) -> None:
    from PIL import Image

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def test_feature_cli_writes_npy_and_paths_json(tmp_path) -> None:
    from pyimgano.feature_cli import main

    _write_png(tmp_path / "a.png", value=0)
    _write_png(tmp_path / "b.png", value=255)

    out_npy = tmp_path / "feats.npy"
    out_json = tmp_path / "paths.json"

    code = main(
        [
            "--root",
            str(tmp_path),
            "--pattern",
            "*.png",
            "--output",
            str(out_npy),
            "--paths-json",
            str(out_json),
            "--extractor",
            "color_hist",
            "--extractor-kwargs",
            json.dumps({"colorspace": "hsv", "bins": [4, 4, 4]}),
        ]
    )
    assert code == 0

    feats = np.load(str(out_npy), allow_pickle=False)
    assert feats.shape == (2, 12)
    assert np.all(np.isfinite(feats))
    paths = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(paths) == 2
