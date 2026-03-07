from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


def _sanitize_stem(stem: str) -> str:
    # Keep filenames portable across filesystems.
    safe = []
    for ch in str(stem):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("._")
    return out or "image"


def _format_regions_suffix(regions: Sequence[Mapping[str, Any]] | None) -> str:
    if not regions:
        return "__r0"

    ids: list[str] = []
    for r in list(regions)[:3]:
        rid = r.get("id", None)
        if rid is None:
            continue
        try:
            ids.append(str(int(rid)))
        except Exception:
            continue
    id_part = "-".join(ids) if ids else "x"

    top = regions[0]
    parts = [f"r{id_part}"]

    smax = top.get("score_max", None)
    if smax is not None:
        try:
            parts.append(f"smax{float(smax):.3f}")
        except Exception:
            pass

    smean = top.get("score_mean", None)
    if smean is not None:
        try:
            parts.append(f"smean{float(smean):.3f}")
        except Exception:
            pass

    area = top.get("area", None)
    if area is not None:
        try:
            parts.append(f"a{int(area)}")
        except Exception:
            pass

    return "__" + "_".join(parts)


def build_overlay_filename(
    *,
    index: int,
    stem: str,
    regions: Sequence[Mapping[str, Any]] | None,
    ext: str = ".png",
) -> str:
    """Build a debug-friendly overlay filename including defect region summary."""

    suffix = _format_regions_suffix(regions)
    safe_stem = _sanitize_stem(stem)
    extension = str(ext) if str(ext).startswith(".") else f".{ext}"
    return f"{int(index):06d}_{safe_stem}{suffix}{extension}"


def save_overlay_image(
    image_path: str | Path,
    *,
    anomaly_map: np.ndarray | None,
    defect_mask: np.ndarray | None,
    out_path: str | Path,
    heatmap_alpha: float = 0.45,
    mask_alpha: float = 0.25,
) -> Path:
    """Save an RGB overlay for false-positive debugging.

    Overlays combine:
    - Original image (RGB)
    - Optional anomaly-map heatmap (JET colormap)
    - Optional defect mask fill + outline
    """

    in_path = Path(image_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as im:
        im_rgb = im.convert("RGB")
        img = np.asarray(im_rgb, dtype=np.uint8)

    h_img, w_img = int(img.shape[0]), int(img.shape[1])
    canvas = img.astype(np.float32)

    if anomaly_map is not None:
        m = np.asarray(anomaly_map, dtype=np.float32)
        if m.ndim != 2:
            raise ValueError(f"anomaly_map must be 2D, got shape {m.shape}")

        vmin = float(np.min(m)) if m.size else 0.0
        vmax = float(np.max(m)) if m.size else 0.0
        if vmax > vmin:
            mn = (m - vmin) / (vmax - vmin)
        else:
            mn = np.zeros_like(m, dtype=np.float32)

        heat_u8 = np.clip(mn * 255.0, 0.0, 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_bgr = cv2.resize(heat_bgr, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        a = float(heatmap_alpha)
        a = float(min(max(a, 0.0), 1.0))
        canvas = (1.0 - a) * canvas + a * heat_rgb

    if defect_mask is not None:
        mask = np.asarray(defect_mask, dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError(f"defect_mask must be 2D, got shape {mask.shape}")
        if mask.shape != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        mask01 = (mask > 0).astype(np.uint8)
        if int(mask01.max()) > 0:
            a = float(mask_alpha)
            a = float(min(max(a, 0.0), 1.0))
            red = np.asarray([255.0, 0.0, 0.0], dtype=np.float32)
            canvas[mask01 > 0] = (1.0 - a) * canvas[mask01 > 0] + a * red

            # Outline via morphological gradient.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(mask01, cv2.MORPH_GRADIENT, kernel)
            canvas[edges > 0] = np.asarray([255.0, 255.0, 255.0], dtype=np.float32)

    out_u8 = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
    # Pillow 13 deprecates the `mode=` argument; HWC uint8 arrays map to "RGB".
    Image.fromarray(out_u8).save(out)
    return out
