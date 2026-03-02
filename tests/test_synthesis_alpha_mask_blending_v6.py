from __future__ import annotations

import numpy as np


def test_synthesizer_uses_alpha_mask_for_alpha_blend() -> None:
    from pyimgano.synthesis.presets import PresetResult
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    def _preset(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
        base = np.asarray(image_u8, dtype=np.uint8)
        overlay = base.copy()
        overlay[0, 0, :] = 200

        # Binary mask says "this pixel is anomalous".
        mask = np.zeros(base.shape[:2], dtype=np.uint8)
        mask[0, 0] = 255

        # Alpha mask carries *continuous* blending strength.
        alpha_mask = np.zeros(base.shape[:2], dtype=np.uint8)
        alpha_mask[0, 0] = 128

        return PresetResult(
            overlay_u8=overlay,
            mask_u8=mask,
            alpha_mask_u8=alpha_mask,
            meta={"preset": "unit"},
        )

    synth = AnomalySynthesizer(
        SynthSpec(preset="unit", probability=1.0, blend="alpha", alpha=1.0),
        preset_fn=_preset,
    )

    img = np.zeros((8, 8, 3), dtype=np.uint8) + 80
    out = synth.synthesize(img, seed=0)
    out_img = np.asarray(out.image_u8, dtype=np.uint8)

    # If the synthesizer ignores `alpha_mask_u8`, binary mask would force the pixel to 200.
    px = int(out_img[0, 0, 0])
    assert 80 < px < 200

