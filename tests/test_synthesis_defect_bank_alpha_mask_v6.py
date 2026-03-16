from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_defect_rgba(bank_dir: Path) -> None:
    import cv2

    bank_dir.mkdir(parents=True, exist_ok=True)

    # BGRA defect image with a soft alpha mask:
    # - 0 outside
    # - 128 in an outer rectangle
    # - 255 in an inner rectangle
    img = np.zeros((24, 32, 4), dtype=np.uint8)
    img[:, :, :3] = (10, 10, 200)  # reddish-ish in BGR
    alpha = np.zeros((24, 32), dtype=np.uint8)
    cv2.rectangle(alpha, (6, 6), (26, 18), color=128, thickness=-1)
    cv2.rectangle(alpha, (10, 9), (22, 15), color=255, thickness=-1)
    img[:, :, 3] = alpha

    cv2.imwrite(str(bank_dir / "defect.png"), img)


def test_defect_bank_preset_exposes_alpha_mask_when_available(tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    from pyimgano.synthesis.defect_bank import DefectBank, make_defect_bank_preset

    bank_dir = tmp_path / "bank"
    _write_defect_rgba(bank_dir)
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

    # Binary mask is still expected for label / manifest semantics.
    mask_u8 = np.asarray(out.mask_u8, dtype=np.uint8)
    uniq = {int(v) for v in np.unique(mask_u8).tolist()}
    assert uniq.issubset({0, 255})

    # But alpha blending can use a continuous alpha mask when provided (e.g. PNG alpha).
    assert getattr(out, "alpha_mask_u8", None) is not None
    alpha_u8 = np.asarray(out.alpha_mask_u8, dtype=np.uint8)
    assert alpha_u8.shape == base.shape[:2]
    assert np.any((alpha_u8 > 0) & (alpha_u8 < 255))
