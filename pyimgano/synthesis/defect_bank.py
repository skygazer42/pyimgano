from __future__ import annotations

"""Defect bank copy/paste preset (industrial synthesis utility).

This module supports a common industrial workflow:
- Build a small library ("bank") of real defect crops and masks.
- Paste/blend them onto normal images to augment anomaly diversity.

Key constraints
---------------
- Bank contents are external data (not shipped in the repo).
- Supports either an explicit mask file or an alpha channel in the defect PNG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .masks import ensure_u8_mask
from .presets import PresetFn, PresetResult


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DefectBankItem:
    image_path: Path
    mask_path: Optional[Path] = None


class DefectBank:
    def __init__(self, items: list[DefectBankItem], *, root: Path) -> None:
        if not items:
            raise ValueError("DefectBank requires at least one item.")
        self.items = list(items)
        self.root = Path(root)

    @classmethod
    def from_dir(cls, defect_bank_dir: str | Path) -> "DefectBank":
        root = Path(defect_bank_dir)
        if not root.exists():
            raise FileNotFoundError(f"defect_bank_dir not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"defect_bank_dir must be a directory: {root}")

        items: list[DefectBankItem] = []
        masks_dir = root / "masks"
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf not in SUPPORTED_EXTENSIONS:
                continue
            if p.stem.endswith("_mask"):
                continue

            mask: Path | None = None
            cand1 = p.with_name(f"{p.stem}_mask{p.suffix}")
            cand2 = p.with_name(f"{p.stem}_mask.png")
            cand3 = masks_dir / f"{p.stem}.png"
            cand4 = masks_dir / f"{p.stem}{p.suffix}"
            for c in (cand1, cand2, cand3, cand4):
                if c.exists() and c.is_file():
                    mask = c
                    break

            items.append(DefectBankItem(image_path=p, mask_path=mask))

        return cls(items, root=root)

    def sample(self, rng: np.random.Generator) -> DefectBankItem:
        idx = int(rng.integers(0, len(self.items)))
        return self.items[idx]


def _as_u8_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.asarray(arr, dtype=np.uint8)
    raise ValueError(f"Expected (H,W) or (H,W,3) uint8 image, got shape={arr.shape}")


def _ensure_u8_mask_values(mask: np.ndarray, *, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != tuple(shape_hw):
        raise ValueError(f"mask must have shape {shape_hw}, got {arr.shape}")

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255

    if arr.dtype == np.uint8:
        return np.asarray(arr, dtype=np.uint8)

    arr_f = arr.astype(np.float32)
    if float(np.max(arr_f)) <= 1.0:
        arr_f = arr_f * 255.0
    return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)


def _load_defect_item_u8(
    item: DefectBankItem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    import cv2  # local import (opencv is a core dependency)

    img_raw = cv2.imread(str(item.image_path), cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Failed to read defect image: {item.image_path}")

    alpha_mask_u8 = None
    if item.mask_path is not None:
        m = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Failed to read defect mask: {item.mask_path}")
        alpha_mask_u8 = np.asarray(m, dtype=np.uint8)

    img_u8: np.ndarray
    if img_raw.ndim == 2:
        img_u8 = cv2.cvtColor(np.asarray(img_raw, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    elif img_raw.ndim == 3 and img_raw.shape[2] == 4:
        bgr = np.asarray(img_raw[:, :, :3], dtype=np.uint8)
        alpha = np.asarray(img_raw[:, :, 3], dtype=np.uint8)
        img_u8 = bgr
        if alpha_mask_u8 is None:
            alpha_mask_u8 = alpha
    elif img_raw.ndim == 3 and img_raw.shape[2] == 3:
        img_u8 = np.asarray(img_raw, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported defect image shape: {img_raw.shape}")

    if alpha_mask_u8 is None:
        # Fallback: if no mask is provided, treat the entire defect crop as active.
        alpha_mask_u8 = np.full(img_u8.shape[:2], 255, dtype=np.uint8)

    if alpha_mask_u8.shape != img_u8.shape[:2]:
        alpha_mask_u8 = cv2.resize(
            alpha_mask_u8,
            (int(img_u8.shape[1]), int(img_u8.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )

    alpha_mask_u8 = _ensure_u8_mask_values(alpha_mask_u8, shape_hw=img_u8.shape[:2])
    mask_u8 = ensure_u8_mask(alpha_mask_u8, shape_hw=img_u8.shape[:2])

    meta: dict[str, object] = {
        "item_path": str(item.image_path),
        "mask_path": (None if item.mask_path is None else str(item.mask_path)),
        "has_alpha_mask": bool(item.mask_path is None and img_raw.ndim == 3 and img_raw.shape[2] == 4),
    }
    return img_u8, mask_u8, alpha_mask_u8, meta


def _crop_to_mask(
    img_u8: np.ndarray,
    alpha_mask_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    m = np.asarray(alpha_mask_u8, dtype=np.uint8) > 0
    if not np.any(m):
        empty = np.zeros_like(alpha_mask_u8, dtype=np.uint8)
        return np.asarray(img_u8, dtype=np.uint8), empty, {"skipped": True, "reason": "empty_mask"}

    ys, xs = np.where(m)
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    crop_img = np.asarray(img_u8[y0:y1, x0:x1], dtype=np.uint8)
    crop_alpha = np.asarray(alpha_mask_u8[y0:y1, x0:x1], dtype=np.uint8)
    return crop_img, crop_alpha, {"crop_xyxy": (int(x0), int(y0), int(x1 - 1), int(y1 - 1))}


def make_defect_bank_preset(
    bank: DefectBank,
    *,
    min_scale: float = 0.18,
    max_scale: float = 0.55,
    max_rotate_deg: float = 18.0,
    flip_prob: float = 0.5,
) -> PresetFn:
    """Create a `PresetFn` that samples and pastes defects from a bank."""

    def _preset(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
        import cv2  # local import (opencv is a core dependency)

        base = _as_u8_bgr(image_u8)
        h, w = int(base.shape[0]), int(base.shape[1])

        item = bank.sample(rng)
        defect_u8, defect_mask_u8, defect_alpha_u8, meta_item = _load_defect_item_u8(item)
        defect_u8, defect_alpha_u8, meta_crop = _crop_to_mask(defect_u8, defect_alpha_u8)
        if meta_crop.get("skipped"):
            empty = np.zeros((h, w), dtype=np.uint8)
            meta = {"preset": "defect_bank", **dict(meta_item), **dict(meta_crop)}
            return PresetResult(overlay_u8=np.asarray(base, dtype=np.uint8), mask_u8=empty, meta=meta)

        # Keep a binary mask for label semantics, but preserve alpha for blending when available.
        defect_mask_u8 = ensure_u8_mask(defect_alpha_u8, shape_hw=defect_u8.shape[:2])

        # Random flips.
        if float(rng.uniform(0.0, 1.0)) < float(flip_prob):
            defect_u8 = np.ascontiguousarray(defect_u8[:, ::-1])
            defect_mask_u8 = np.ascontiguousarray(defect_mask_u8[:, ::-1])
            defect_alpha_u8 = np.ascontiguousarray(defect_alpha_u8[:, ::-1])
            flip_h = True
        else:
            flip_h = False
        if float(rng.uniform(0.0, 1.0)) < float(flip_prob):
            defect_u8 = np.ascontiguousarray(defect_u8[::-1, :])
            defect_mask_u8 = np.ascontiguousarray(defect_mask_u8[::-1, :])
            defect_alpha_u8 = np.ascontiguousarray(defect_alpha_u8[::-1, :])
            flip_v = True
        else:
            flip_v = False

        # Random rotation around center.
        angle = float(rng.uniform(-abs(float(max_rotate_deg)), abs(float(max_rotate_deg))))
        if abs(angle) > 1e-3:
            dh, dw = int(defect_u8.shape[0]), int(defect_u8.shape[1])
            center = (float(dw - 1) * 0.5, float(dh - 1) * 0.5)
            m = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
            defect_u8 = cv2.warpAffine(
                defect_u8,
                m,
                dsize=(dw, dh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            defect_alpha_u8 = cv2.warpAffine(
                defect_alpha_u8,
                m,
                dsize=(dw, dh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            defect_alpha_u8 = _ensure_u8_mask_values(defect_alpha_u8, shape_hw=defect_u8.shape[:2])
            defect_mask_u8 = ensure_u8_mask(defect_alpha_u8, shape_hw=defect_u8.shape[:2])

        # Resize defect crop to a fraction of the base image.
        scale = float(rng.uniform(float(min_scale), float(max_scale)))
        scale = float(np.clip(scale, 0.05, 0.95))
        target_max = float(min(h, w)) * scale
        dh, dw = int(defect_u8.shape[0]), int(defect_u8.shape[1])
        denom = float(max(dh, dw))
        if denom <= 0.0:
            empty = np.zeros((h, w), dtype=np.uint8)
            meta = {"preset": "defect_bank", **dict(meta_item), "skipped": True, "reason": "bad_defect_shape"}
            return PresetResult(overlay_u8=np.asarray(base, dtype=np.uint8), mask_u8=empty, meta=meta)

        r = float(target_max / denom)
        new_h = int(np.clip(int(round(dh * r)), 1, h))
        new_w = int(np.clip(int(round(dw * r)), 1, w))
        defect_r = cv2.resize(defect_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha_r = cv2.resize(defect_alpha_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha_r = _ensure_u8_mask_values(alpha_r, shape_hw=(new_h, new_w))
        mask_r = ensure_u8_mask(alpha_r, shape_hw=(new_h, new_w))

        if (new_h >= h) or (new_w >= w):
            y0 = 0
            x0 = 0
        else:
            y0 = int(rng.integers(0, h - new_h + 1))
            x0 = int(rng.integers(0, w - new_w + 1))

        overlay = np.asarray(base, dtype=np.uint8).copy()
        overlay[y0 : y0 + new_h, x0 : x0 + new_w] = np.asarray(defect_r, dtype=np.uint8)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0 : y0 + new_h, x0 : x0 + new_w] = np.asarray(mask_r, dtype=np.uint8)

        full_alpha = np.zeros((h, w), dtype=np.uint8)
        full_alpha[y0 : y0 + new_h, x0 : x0 + new_w] = np.asarray(alpha_r, dtype=np.uint8)

        try:
            rel_item = str(Path(item.image_path).resolve().relative_to(bank.root.resolve()))
        except Exception:
            rel_item = str(item.image_path)

        meta: dict[str, object] = {
            "preset": "defect_bank",
            "defect_item": rel_item,
            "scale": float(scale),
            "angle_deg": float(angle),
            "flip_h": bool(flip_h),
            "flip_v": bool(flip_v),
            "bbox_xyxy": (int(x0), int(y0), int(x0 + new_w - 1), int(y0 + new_h - 1)),
        }
        meta.update(dict(meta_item))
        meta.update(dict(meta_crop))

        return PresetResult(
            overlay_u8=np.asarray(overlay, dtype=np.uint8),
            mask_u8=full_mask,
            meta=meta,
            alpha_mask_u8=full_alpha,
        )

    return _preset


__all__ = ["DefectBank", "DefectBankItem", "make_defect_bank_preset"]
