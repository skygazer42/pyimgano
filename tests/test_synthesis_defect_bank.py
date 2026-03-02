from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_defect_pair(bank_dir: Path) -> None:
    import cv2

    bank_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[:, :] = (10, 10, 200)  # reddish-ish in BGR

    mask = np.zeros((24, 32), dtype=np.uint8)
    cv2.rectangle(mask, (6, 6), (26, 18), color=255, thickness=-1)

    cv2.imwrite(str(bank_dir / "defect.png"), img)
    cv2.imwrite(str(bank_dir / "defect_mask.png"), mask)


def test_defect_bank_preset_pastes_nonempty_mask(tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    from pyimgano.synthesis.defect_bank import DefectBank, make_defect_bank_preset

    bank_dir = tmp_path / "bank"
    _write_defect_pair(bank_dir)
    bank = DefectBank.from_dir(bank_dir)

    preset = make_defect_bank_preset(
        bank,
        min_scale=0.35,
        max_scale=0.35,
        max_rotate_deg=0.0,
        flip_prob=0.0,
    )

    base = np.zeros((64, 64, 3), dtype=np.uint8) + 80
    rng = np.random.default_rng(0)
    out = preset(base, rng)

    assert out.overlay_u8.shape == base.shape
    assert out.overlay_u8.dtype == np.uint8
    assert out.mask_u8.shape == base.shape[:2]
    assert out.mask_u8.dtype == np.uint8
    assert int(np.sum(out.mask_u8 > 0)) > 0

    meta = dict(out.meta)
    assert meta.get("preset") == "defect_bank"
    assert "defect_item" in meta
    assert "bbox_xyxy" in meta

