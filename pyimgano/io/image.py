from __future__ import annotations

from pathlib import Path
from typing import Literal


ColorMode = Literal["bgr", "rgb", "gray"]


def read_image(path: str | Path, *, color: ColorMode = "bgr"):
    """Read an image from disk via OpenCV.

    Parameters
    ----------
    path:
        Image file path.
    color:
        - "bgr": default OpenCV color ordering (H,W,3)
        - "rgb": converted to RGB (H,W,3)
        - "gray": single channel (H,W)
    """

    import cv2

    path_str = str(path)
    if color == "gray":
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path_str, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path_str}")

    if color == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if color == "bgr":
        return img
    if color == "gray":
        return img

    raise ValueError(f"Unknown color mode: {color!r}. Choose from: bgr, rgb, gray.")


def convert_color(image, *, src: ColorMode, dst: ColorMode):
    """Convert between common image color modes."""

    import cv2

    if src == dst:
        return image

    if src == "bgr" and dst == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if src == "rgb" and dst == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if src == "bgr" and dst == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if src == "rgb" and dst == "gray":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if src == "gray" and dst == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if src == "gray" and dst == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    raise ValueError(f"Unsupported conversion: {src!r} -> {dst!r}.")


def resize_image(
    image,
    size_hw: tuple[int, int],
    *,
    is_mask: bool = False,
):
    """Resize an image/mask to (H,W) using OpenCV.

    Notes
    -----
    OpenCV uses (W,H) order for its `dsize` argument, while most of `pyimgano`
    uses (H,W). This helper standardizes on (H,W).
    """

    import cv2

    h, w = int(size_hw[0]), int(size_hw[1])
    interp = cv2.INTER_NEAREST if bool(is_mask) else cv2.INTER_AREA
    return cv2.resize(image, (w, h), interpolation=interp)

