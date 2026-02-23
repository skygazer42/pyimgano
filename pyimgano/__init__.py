"""PyImgAno - Enterprise-Grade Visual Anomaly Detection Toolkit.

Keep top-level imports lightweight: many submodules depend on optional heavy
deps (torch, cv2, sklearn, etc.). We lazy-load these exports on demand so that
`import pyimgano` and `import pyimgano.cli` work in minimal environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.6.1"

__all__ = [
    # Modules
    "datasets",
    "detectors",
    "inputs",
    "inference",
    "models",
    "pipelines",
    "preprocessing",
    "utils",
    "visualization",
    # Evaluation
    "evaluate_detector",
    "compute_auroc",
    "compute_average_precision",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "print_evaluation_summary",
    # Benchmark
    "AlgorithmBenchmark",
    "quick_benchmark",
]


_LAZY_SUBMODULES = {
    "datasets",
    "detectors",
    "inputs",
    "inference",
    "models",
    "pipelines",
    "preprocessing",
    "utils",
    "visualization",
}

_LAZY_EXPORTS = {
    # Benchmark
    "AlgorithmBenchmark": ("benchmark", "AlgorithmBenchmark"),
    "quick_benchmark": ("benchmark", "quick_benchmark"),
    # Evaluation
    "evaluate_detector": ("evaluation", "evaluate_detector"),
    "compute_auroc": ("evaluation", "compute_auroc"),
    "compute_average_precision": ("evaluation", "compute_average_precision"),
    "compute_classification_metrics": ("evaluation", "compute_classification_metrics"),
    "find_optimal_threshold": ("evaluation", "find_optimal_threshold"),
    "print_evaluation_summary": ("evaluation", "print_evaluation_summary"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attr = target
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tooling convenience
    return sorted(set(globals()) | set(__all__))
