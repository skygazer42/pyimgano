from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
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


def _safe_run(cmd: list[str], *, timeout_s: float = 1.5) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=float(timeout_s),
        )
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    return out if out else None


def _collect_git_info() -> dict[str, Any] | None:
    """Best-effort git metadata for source checkouts.

    Notes
    -----
    When `pyimgano` is installed from a wheel/sdist, callers are often outside
    a git checkout. This must remain best-effort and never raise.
    """

    if shutil.which("git") is None:
        return None

    inside = _safe_run(["git", "rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        return None

    commit = _safe_run(["git", "rev-parse", "HEAD"])
    branch = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    describe = _safe_run(["git", "describe", "--tags", "--always", "--dirty"])

    dirty = None
    status = _safe_run(["git", "status", "--porcelain"])
    if status is not None:
        dirty = bool(status.strip())

    return {
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
    }


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
        "pip": _dist_version("pip"),
        "torch": _dist_version("torch"),
        "torchvision": _dist_version("torchvision"),
        # Optional runtimes/backends (best-effort).
        "onnxruntime": _dist_version("onnxruntime"),
        "onnx": _dist_version("onnx"),
        "onnxscript": _dist_version("onnxscript"),
        "openvino": _dist_version("openvino"),
        "scikit_image": _dist_version("scikit-image"),
        "numba": _dist_version("numba"),
        "open_clip_torch": _dist_version("open-clip-torch"),
        "anomalib": _dist_version("anomalib"),
        # faiss dist naming varies across platforms.
        "faiss_cpu": _dist_version("faiss-cpu") or _dist_version("faiss"),
    }

    try:
        from pyimgano.utils.extras import EXTRA_ROOT_MODULES, extra_installed

        extras = {str(e): bool(extra_installed(str(e))) for e in EXTRA_ROOT_MODULES.keys()}
    except Exception:
        extras = None

    git_info = _collect_git_info()
    fingerprint_src = {
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "extras_installed": extras,
        "git": (
            {"commit": git_info.get("commit"), "describe": git_info.get("describe")}
            if git_info
            else None
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_src, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()

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
        "fingerprint_sha256": fingerprint,
        "git": git_info,
        "packages": packages,
        "extras_installed": extras,
        "torch": torch_info,
    }
