from __future__ import annotations

import numpy as np
import pytest


def test_camera_artifacts_defocus_and_lens_distortion_primitives() -> None:
    pytest.importorskip("cv2")

    from pyimgano.synthesis.camera_artifacts import apply_defocus_blur, apply_lens_distortion

    rng = np.random.default_rng(0)
    img = (rng.integers(0, 255, size=(64, 64, 3))).astype(np.uint8)

    ov1, m1, meta1 = apply_defocus_blur(img, rng=rng, strength=1.0)
    assert ov1.shape == img.shape
    assert ov1.dtype == np.uint8
    assert m1.shape == img.shape[:2]
    assert m1.dtype == np.uint8
    assert int(np.min(m1)) == 255 and int(np.max(m1)) == 255
    assert meta1.get("mode") == "defocus"
    assert float(meta1.get("strength", 0.0)) >= 0.0

    ov2, m2, meta2 = apply_lens_distortion(img, rng=rng, strength=1.0)
    assert ov2.shape == img.shape
    assert ov2.dtype == np.uint8
    assert m2.shape == img.shape[:2]
    assert m2.dtype == np.uint8
    assert int(np.min(m2)) == 255 and int(np.max(m2)) == 255
    assert meta2.get("mode") == "lens_distortion"
    assert "k1" in meta2


def test_camera_artifacts_presets_registered() -> None:
    pytest.importorskip("cv2")

    from pyimgano.synthesis.presets import make_preset

    rng = np.random.default_rng(0)
    img = np.zeros((32, 40, 3), dtype=np.uint8)

    pr_defocus = make_preset("defocus")(img, rng)
    assert pr_defocus.overlay_u8.shape == img.shape
    assert pr_defocus.mask_u8.shape == img.shape[:2]
    assert pr_defocus.meta.get("preset") == "defocus"

    pr_lens = make_preset("lens_distortion")(img, rng)
    assert pr_lens.overlay_u8.shape == img.shape
    assert pr_lens.mask_u8.shape == img.shape[:2]
    assert pr_lens.meta.get("preset") == "lens_distortion"
