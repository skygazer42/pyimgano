from __future__ import annotations

from typing import Any, Protocol

from pyimgano.workbench.config import WorkbenchConfig


class Recipe(Protocol):
    def __call__(self, config: WorkbenchConfig) -> dict[str, Any]:
        """Execute a workbench run and return a JSON-friendly report payload."""

