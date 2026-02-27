from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from .blend import alpha_blend, poisson_blend
from .masks import apply_roi_mask, ensure_u8_mask
from .presets import PresetFn, make_preset


_BlendMode = Literal["alpha", "poisson"]


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")


@dataclass(frozen=True)
class SynthSpec:
    """Configuration for `AnomalySynthesizer`."""

    preset: str = "scratch"
    probability: float = 1.0
    blend: _BlendMode = "alpha"
    alpha: float = 0.85
    poisson_mode: str = "normal"  # normal|mixed
    max_tries: int = 5


@dataclass(frozen=True)
class SynthResult:
    image_u8: np.ndarray
    mask_u8: np.ndarray
    label: int
    meta: dict[str, Any]


class AnomalySynthesizer:
    """Configurable synthetic anomaly generator.

    Notes
    -----
    - This utility is intended for **industrial iteration** and robustness.
    - It is NOT a replacement for real defects.
    """

    def __init__(self, spec: SynthSpec | None = None, *, preset_fn: Optional[PresetFn] = None) -> None:
        self.spec = SynthSpec() if spec is None else spec
        self._preset_fn = make_preset(self.spec.preset) if preset_fn is None else preset_fn

    def make_rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(int(seed))

    def synthesize(
        self,
        image_u8: np.ndarray,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> SynthResult:
        """Generate a synthetic anomaly and mask.

        Parameters
        ----------
        image_u8:
            Input image (uint8).
        seed / rng:
            Deterministic control. If `rng` is provided it is used directly.
        roi_mask:
            Optional ROI mask (H,W) where non-zero indicates allowed anomaly region.
        """

        img = _as_u8_image(image_u8)
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = self.make_rng(int(seed))

        p = float(self.spec.probability)
        if p <= 0.0:
            empty = np.zeros(img.shape[:2], dtype=np.uint8)
            return SynthResult(image_u8=np.asarray(img, dtype=np.uint8), mask_u8=empty, label=0, meta={"skipped": True})
        if p < 1.0:
            if float(rng.uniform(0.0, 1.0)) > p:
                empty = np.zeros(img.shape[:2], dtype=np.uint8)
                return SynthResult(
                    image_u8=np.asarray(img, dtype=np.uint8),
                    mask_u8=empty,
                    label=0,
                    meta={"skipped": True, "reason": "probability"},
                )

        tries = max(1, int(self.spec.max_tries))
        last_meta: dict[str, Any] = {"preset": str(self.spec.preset)}
        for _ in range(tries):
            pr = self._preset_fn(img, rng)
            overlay = _as_u8_image(pr.overlay_u8)
            mask = ensure_u8_mask(pr.mask_u8, shape_hw=img.shape[:2])
            mask = apply_roi_mask(mask, roi_mask)

            if int(np.sum(mask > 0)) == 0:
                last_meta = dict(pr.meta)
                continue

            blend = str(self.spec.blend).lower().strip()
            if blend == "poisson":
                out_img = poisson_blend(
                    img,
                    overlay,
                    mask,
                    mode="mixed" if str(self.spec.poisson_mode).lower().strip() == "mixed" else "normal",
                )
            elif blend == "alpha":
                out_img = alpha_blend(img, overlay, mask, alpha=float(self.spec.alpha))
            else:
                raise ValueError(f"Unknown blend mode: {self.spec.blend!r}")

            meta = dict(pr.meta)
            meta.update({"blend": blend})
            return SynthResult(
                image_u8=np.asarray(out_img, dtype=np.uint8),
                mask_u8=np.asarray(mask, dtype=np.uint8),
                label=1,
                meta=meta,
            )

        # If ROI constraint prevents any non-empty mask, return original.
        empty = np.zeros(img.shape[:2], dtype=np.uint8)
        meta = dict(last_meta)
        meta.update({"skipped": True, "reason": "empty_mask"})
        return SynthResult(image_u8=np.asarray(img, dtype=np.uint8), mask_u8=empty, label=0, meta=meta)

    def __call__(
        self,
        image_u8: np.ndarray,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> SynthResult:
        return self.synthesize(image_u8, seed=seed, rng=rng, roi_mask=roi_mask)

