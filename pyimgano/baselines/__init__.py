"""Baseline suites for industrial model selection.

This package defines curated *benchmark suites* (collections of model presets)
that can be executed via `pyimgano-benchmark --suite ...`.

Design goals:
- import-light: suite discovery must not import optional heavy deps (torch, etc.)
- reproducible: suite entries resolve to explicit (model, kwargs) pairs
"""

from __future__ import annotations

from .suites import (
    Baseline,
    BaselineSuite,
    get_baseline_suite,
    list_baseline_suites,
    resolve_suite_baselines,
)

__all__ = [
    "Baseline",
    "BaselineSuite",
    "get_baseline_suite",
    "list_baseline_suites",
    "resolve_suite_baselines",
]
