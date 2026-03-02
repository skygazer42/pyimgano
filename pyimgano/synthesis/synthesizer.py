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
    num_defects: int = 1
    severity: float = 1.0  # [0,1] global strength scalar
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
        num_defects: Optional[int] = None,
        severity: Optional[float] = None,
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
        num_defects:
            Optional override for the number of defect injections to attempt per image.
        severity:
            Optional override for the global defect strength scalar in [0,1].
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
        target_defects = int(self.spec.num_defects if num_defects is None else num_defects)
        target_defects = max(1, target_defects)

        sev = float(self.spec.severity if severity is None else severity)
        sev = float(np.clip(sev, 0.0, 1.0))

        blend = str(self.spec.blend).lower().strip()
        poisson_mode = "mixed" if str(self.spec.poisson_mode).lower().strip() == "mixed" else "normal"

        out_img = np.asarray(img, dtype=np.uint8)
        union_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        last_meta: dict[str, Any] = {"preset": str(self.spec.preset)}
        defects_meta: list[dict[str, Any]] = []

        for defect_idx in range(target_defects):
            applied = False
            for attempt_idx in range(tries):
                pr = self._preset_fn(out_img, rng)
                overlay = _as_u8_image(pr.overlay_u8)
                mask = ensure_u8_mask(pr.mask_u8, shape_hw=out_img.shape[:2])
                mask = apply_roi_mask(mask, roi_mask)

                if int(np.sum(mask > 0)) == 0:
                    last_meta = dict(pr.meta)
                    continue

                if blend == "poisson":
                    # Poisson has no alpha knob; scale the overlay towards base first.
                    if sev < 1.0:
                        o = overlay.astype(np.float32)
                        b = out_img.astype(np.float32)
                        overlay = np.clip(b + (o - b) * sev, 0.0, 255.0).astype(np.uint8)
                    out_img = poisson_blend(out_img, overlay, mask, mode=poisson_mode)
                elif blend == "alpha":
                    alpha_mask = getattr(pr, "alpha_mask_u8", None)
                    if alpha_mask is not None:
                        alpha_arr = np.asarray(alpha_mask)
                        if alpha_arr.shape != out_img.shape[:2]:
                            raise ValueError(
                                "alpha_mask_u8 must have shape (H,W) matching the image. "
                                f"Got {alpha_arr.shape} for image {out_img.shape}."
                            )
                        if alpha_arr.dtype != np.uint8:
                            a = alpha_arr.astype(np.float32)
                            if float(np.max(a)) <= 1.0:
                                a = a * 255.0
                            alpha_arr = np.clip(a, 0.0, 255.0).astype(np.uint8)

                        # Ensure ROI constraint applies to the blend alpha as well.
                        alpha_arr = np.asarray(alpha_arr, dtype=np.uint8)
                        alpha_arr = alpha_arr * (mask > 0).astype(np.uint8)
                        blend_mask = alpha_arr
                    else:
                        blend_mask = mask

                    eff_alpha = float(np.clip(float(self.spec.alpha) * sev, 0.0, 1.0))
                    out_img = alpha_blend(out_img, overlay, blend_mask, alpha=eff_alpha)
                else:
                    raise ValueError(f"Unknown blend mode: {self.spec.blend!r}")

                union_mask = np.maximum(union_mask, np.asarray(mask, dtype=np.uint8))
                m = dict(pr.meta)
                m.update({"defect_index": int(defect_idx), "attempt_index": int(attempt_idx)})
                defects_meta.append(m)
                applied = True
                break

            if not applied:
                break

        if defects_meta:
            meta = dict(defects_meta[-1])
            meta.update(
                {
                    "blend": blend,
                    "severity": float(sev),
                    "num_defects": int(target_defects),
                    "defects_applied": int(len(defects_meta)),
                }
            )
            # Stable "preset id" field for downstream dataset/meta consumers.
            preset_id = meta.get("preset", None)
            meta["preset_id"] = str(self.spec.preset) if preset_id is None else str(preset_id)
            if len(defects_meta) > 1:
                meta["defects"] = list(defects_meta)

            return SynthResult(
                image_u8=np.asarray(out_img, dtype=np.uint8),
                mask_u8=np.asarray(union_mask, dtype=np.uint8),
                label=1,
                meta=meta,
            )

        # If ROI constraint prevents any non-empty mask, return original.
        empty = np.zeros(img.shape[:2], dtype=np.uint8)
        meta = dict(last_meta)
        meta.update(
            {
                "skipped": True,
                "reason": "empty_mask",
                "blend": blend,
                "severity": float(sev),
                "num_defects": int(target_defects),
                "defects_applied": 0,
                "preset_id": str(self.spec.preset),
            }
        )
        return SynthResult(image_u8=np.asarray(img, dtype=np.uint8), mask_u8=empty, label=0, meta=meta)

    def __call__(
        self,
        image_u8: np.ndarray,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        roi_mask: Optional[np.ndarray] = None,
        num_defects: Optional[int] = None,
        severity: Optional[float] = None,
    ) -> SynthResult:
        return self.synthesize(
            image_u8,
            seed=seed,
            rng=rng,
            roi_mask=roi_mask,
            num_defects=num_defects,
            severity=severity,
        )
