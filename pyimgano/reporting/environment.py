from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from pyimgano.utils.optional_deps import optional_import


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect lightweight, JSON-friendly environment metadata for run artifacts."""

    torch, _torch_error = optional_import("torch")

    torch_info: dict[str, Any] = {"available": bool(torch is not None)}
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        except Exception:
            cuda_available = False
        torch_info.update(
            {
                "version": getattr(torch, "__version__", None),
                "cuda_available": cuda_available,
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            }
        )
    else:
        torch_info["error"] = str(_torch_error) if _torch_error is not None else None

    packages = {
        "pyimgano": _dist_version("pyimgano"),
        "numpy": _dist_version("numpy"),
        "opencv_python": _dist_version("opencv-python"),
        "scikit_learn": _dist_version("scikit-learn"),
        "torch": _dist_version("torch"),
        "torchvision": _dist_version("torchvision"),
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch": torch_info,
    }
