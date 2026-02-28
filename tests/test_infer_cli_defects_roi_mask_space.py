from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_infer_cli_roi_applies_to_regions_but_masks_can_be_full_or_roi(
    tmp_path: Path,
) -> None:
    from pyimgano.infer_cli import main as infer_main
    from pyimgano.models.registry import MODEL_REGISTRY

    amap = np.zeros((10, 10), dtype=np.float32)
    amap[0:2, 0:2] = 1.0  # outside ROI blob
    amap[5:7, 5:7] = 1.0  # inside ROI blob

    class _DummyPixelMapDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def decision_function(self, X):  # noqa: ANN001
            return np.zeros((len(list(X)),), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001
            return np.asarray(amap, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_defects_roi_mask_space_dummy",
        _DummyPixelMapDetector,
        tags=("vision", "classical", "pixel_map"),
        overwrite=True,
    )

    img = tmp_path / "x.png"
    _write_rgb(img)

    masks_full = tmp_path / "masks_full"
    out_full = tmp_path / "out_full.jsonl"
    rc_full = infer_main(
        [
            "--model",
            "test_defects_roi_mask_space_dummy",
            "--input",
            str(img),
            "--defects",
            "--pixel-threshold",
            "0.5",
            "--pixel-threshold-strategy",
            "fixed",
            "--save-masks",
            str(masks_full),
            "--defects-mask-space",
            "full",
            "--roi-xyxy-norm",
            "0.4",
            "0.4",
            "1.0",
            "1.0",
            "--save-jsonl",
            str(out_full),
        ]
    )
    assert rc_full == 0

    masks_roi = tmp_path / "masks_roi"
    out_roi = tmp_path / "out_roi.jsonl"
    rc_roi = infer_main(
        [
            "--model",
            "test_defects_roi_mask_space_dummy",
            "--input",
            str(img),
            "--defects",
            "--pixel-threshold",
            "0.5",
            "--pixel-threshold-strategy",
            "fixed",
            "--save-masks",
            str(masks_roi),
            "--defects-mask-space",
            "roi",
            "--roi-xyxy-norm",
            "0.4",
            "0.4",
            "1.0",
            "1.0",
            "--save-jsonl",
            str(out_roi),
        ]
    )
    assert rc_roi == 0

    rec_full = json.loads(out_full.read_text(encoding="utf-8").strip().splitlines()[0])
    rec_roi = json.loads(out_roi.read_text(encoding="utf-8").strip().splitlines()[0])

    # ROI should always apply to region extraction.
    assert len(rec_full["defects"]["regions"]) == 1
    assert len(rec_roi["defects"]["regions"]) == 1

    mask_full_path = Path(rec_full["defects"]["mask"]["path"])
    mask_roi_path = Path(rec_roi["defects"]["mask"]["path"])
    m_full = np.asarray(Image.open(mask_full_path).convert("L"))
    m_roi = np.asarray(Image.open(mask_roi_path).convert("L"))

    assert int(m_full[0, 0]) == 255  # full mask includes outside ROI blob
    assert int(m_roi[0, 0]) == 0  # ROI mask excludes outside ROI blob
    assert int(np.sum(m_full > 0)) > int(np.sum(m_roi > 0))

