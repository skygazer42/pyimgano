"""Optional dependency helpers.

The core `pyimgano` install should stay lightweight. Optional backends and
accelerators (e.g. `anomalib`, `faiss`) are supported via extras.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional, Tuple


_PIP_NAME_OVERRIDES = {
    # Common module ↔ pip package mismatches.
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "open_clip": "open_clip_torch",
    "faiss": "faiss-cpu",
    "mamba_ssm": "mamba-ssm",
    # Not on PyPI: patchcore-inspection installs as `patchcore`.
    "patchcore": "patchcore @ git+https://github.com/amazon-science/patchcore-inspection.git",
}


def optional_import(module_name: str) -> Tuple[Optional[ModuleType], Optional[BaseException]]:
    """Attempt to import a module, returning (module, error)."""

    try:
        return import_module(module_name), None
    except Exception as exc:  # noqa: BLE001 - return import error without swallowing BaseException
        return None, exc


def require(module_name: str, *, extra: Optional[str] = None, purpose: Optional[str] = None) -> ModuleType:
    """Import `module_name`, raising a clean ImportError with install hint if missing."""

    module, error = optional_import(module_name)
    if module is not None:
        return module

    hint = None
    if extra:
        hint = f"pip install 'pyimgano[{extra}]'"
    else:
        root = str(module_name).split(".", 1)[0]
        pip_target = _PIP_NAME_OVERRIDES.get(root, root)
        hint = f"pip install '{pip_target}'"

    context = f" for {purpose}" if purpose else ""
    raise ImportError(
        f"Optional dependency '{module_name}' is required{context}.\n"
        f"Install it via:\n  {hint}\n"
        f"Original error: {error}"
    ) from error
