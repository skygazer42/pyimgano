"""Plugin loading via Python entry points.

This module exists to close a common gap vs "top-tier" ML toolkits: allowing
third-party packages to extend `pyimgano` without modifying this repo.

Design constraints
------------------
- **Opt-in** by default: importing `pyimgano` must not auto-import plugins.
- **Import-light**: this module must not import heavy optional deps.
- **Actionable errors**: failures should show which entry point broke.

Entry point groups
------------------
This module loads callables from Python entry points. The default group is:

- ``pyimgano.plugins``: generic plugin initializers

Plugins should expose a zero-argument callable that registers models/features
using `pyimgano.models.registry.register_model` / `pyimgano.features.registry.register_feature_extractor`.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Sequence

import importlib.metadata as md


OnError = Literal["raise", "warn", "ignore"]


@dataclass(frozen=True)
class PluginLoadResult:
    group: str
    name: str
    value: str
    status: Literal["loaded", "skipped", "error"]
    error: str | None = None


_LOADED: set[tuple[str, str, str]] = set()


def _entry_points_for_group(group: str) -> list[md.EntryPoint]:
    eps = md.entry_points()
    # Python 3.10+ returns an EntryPoints object with `.select`.
    select = getattr(eps, "select", None)
    if callable(select):
        return list(select(group=str(group)))

    # Older/other: may be a mapping-like structure.
    get = getattr(eps, "get", None)
    if callable(get):
        return list(get(str(group), []))

    # Last resort: treat as an iterable of EntryPoint objects.
    out: list[md.EntryPoint] = []
    for ep in eps:  # type: ignore[assignment]
        if getattr(ep, "group", None) == str(group):
            out.append(ep)
    return out


def load_plugins(
    *,
    groups: Sequence[str] = ("pyimgano.plugins",),
    on_error: OnError = "warn",
) -> list[dict[str, Any]]:
    """Load plugins from entry points and return JSON-friendly results.

    Parameters
    ----------
    groups:
        Entry point groups to load. Defaults to ("pyimgano.plugins",).
    on_error:
        - "raise": raise the first exception
        - "warn": emit a warning and continue (default)
        - "ignore": silently continue
    """

    mode = str(on_error).strip().lower()
    if mode not in {"raise", "warn", "ignore"}:
        raise ValueError("on_error must be one of: raise|warn|ignore")

    results: list[PluginLoadResult] = []
    for group in groups:
        for ep in _entry_points_for_group(str(group)):
            key = (str(ep.group), str(ep.name), str(ep.value))
            if key in _LOADED:
                results.append(
                    PluginLoadResult(
                        group=str(ep.group),
                        name=str(ep.name),
                        value=str(ep.value),
                        status="skipped",
                        error=None,
                    )
                )
                continue

            try:
                obj = ep.load()
                if not callable(obj):
                    raise TypeError(
                        f"Entry point {ep.name!r} in group {ep.group!r} did not resolve to a callable: {obj!r}"
                    )
                obj()
                _LOADED.add(key)
                results.append(
                    PluginLoadResult(
                        group=str(ep.group),
                        name=str(ep.name),
                        value=str(ep.value),
                        status="loaded",
                        error=None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - plugin boundary
                msg = f"Failed to load plugin entry point {ep.name!r} ({ep.value}) in group {ep.group!r}: {exc}"
                results.append(
                    PluginLoadResult(
                        group=str(ep.group),
                        name=str(ep.name),
                        value=str(ep.value),
                        status="error",
                        error=str(exc),
                    )
                )
                if mode == "raise":
                    raise
                if mode == "warn":
                    warnings.warn(msg, RuntimeWarning)
                # ignore => do nothing

    return [asdict(r) for r in results]


__all__ = [
    "PluginLoadResult",
    "load_plugins",
]

