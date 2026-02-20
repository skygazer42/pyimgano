"""Optional dependency helpers.

The core `pyimgano` install should stay lightweight. Optional backends and
accelerators (e.g. `anomalib`, `faiss`) are supported via extras.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional, Tuple


def optional_import(module_name: str) -> Tuple[Optional[ModuleType], Optional[BaseException]]:
    """Attempt to import a module, returning (module, error)."""

    try:
        return import_module(module_name), None
    except BaseException as exc:  # noqa: BLE001 - intentional: return import error
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
        hint = f"pip install '{module_name}'"

    context = f" for {purpose}" if purpose else ""
    raise ImportError(
        f"Optional dependency '{module_name}' is required{context}.\n"
        f"Install it via:\n  {hint}\n"
        f"Original error: {error}"
    ) from error

