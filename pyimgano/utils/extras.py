"""Extras helpers (import-light).

`pyimgano` keeps the core install lightweight and pushes optional runtimes
(torch/onnx/openvino/...) behind `pip install "pyimgano[...]"` extras.

This module centralizes the mapping from *extra name* → *root modules* so that:
- suite skip hints stay consistent with packaging metadata
- `pyimgano-doctor --require-extras ...` can validate environments
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.util import find_spec

from pyimgano.utils.optional_deps import optional_import

# Keep in sync with `pyproject.toml` extras.
EXTRA_ROOT_MODULES: dict[str, tuple[str, ...]] = {
    # Base extras
    "torch": ("torch", "torchvision"),
    "onnx": ("onnxruntime", "onnx", "onnxscript"),
    "openvino": ("openvino",),
    "skimage": ("skimage",),
    "numba": ("numba",),
    "faiss": ("faiss",),
    # Extras that imply torch.
    "clip": ("open_clip", "torch"),
    "anomalib": ("anomalib", "torch"),
    "mamba": ("mamba_ssm", "torch"),
    # Meta-extras (allow `pyimgano-doctor --require-extras backends` style checks).
    "backends": ("anomalib", "faiss", "open_clip", "torch", "torchvision"),
}


def extra_roots(extra: str) -> tuple[str, ...]:
    """Return the list of root modules that define an extra.

    If `extra` is unknown, treat it as a root module name for best-effort checks.
    """

    key = str(extra)
    return EXTRA_ROOT_MODULES.get(key, (key,))


def can_find_root(module_root: str) -> bool:
    """Best-effort root-module existence check (no import side effects)."""

    try:
        return find_spec(str(module_root)) is not None
    except Exception:
        return False


def can_import_root(module_root: str) -> bool:
    """Best-effort import check (catches broken wheels / missing shared libs)."""

    mod, _err = optional_import(str(module_root))
    return bool(mod is not None)


def extra_installed(extra: str) -> bool:
    """Return True when all root modules for `extra` are present (find_spec)."""

    return all(can_find_root(r) for r in extra_roots(str(extra)))


def extra_importable(extra: str) -> bool:
    """Return True when all root modules for `extra` are importable."""

    return all(can_import_root(r) for r in extra_roots(str(extra)))


def extras_install_hint(extras: Iterable[str]) -> str:
    spec = ",".join(sorted({str(e) for e in extras if str(e).strip()}))
    return f"pip install 'pyimgano[{spec}]'"
