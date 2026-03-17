"""PyImgAno - Enterprise-Grade Visual Anomaly Detection Toolkit.

Keep top-level imports lightweight: many submodules depend on optional heavy
deps (torch, cv2, sklearn, etc.). We lazy-load these exports on demand so that
`import pyimgano` and `import pyimgano.cli` work in minimal environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.6.38"

_MODULE_EXPORTS = (
    "datasets",
    "detectors",
    "features",
    "inputs",
    "inference",
    "models",
    "plugins",
    "pipelines",
    "preprocessing",
    "synthesis",
    "utils",
    "visualization",
)
_BENCHMARK_MODULE = "benchmark"
_BENCHMARK_EXPORTS = ("AlgorithmBenchmark", "quick_benchmark")
_EVALUATION_MODULE = "evaluation"
_EVALUATION_EXPORTS = (
    "evaluate_detector",
    "compute_auroc",
    "compute_average_precision",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "print_evaluation_summary",
)
_SYNTHESIS_MODULE = "synthesis.synthesizer"
_SYNTHESIS_EXPORTS = ("AnomalySynthesizer", "SynthSpec", "SynthResult")

__all__ = [
    *_MODULE_EXPORTS,
    *_EVALUATION_EXPORTS,
    *_BENCHMARK_EXPORTS,
    *_SYNTHESIS_EXPORTS,
]


_LAZY_SUBMODULES = set(_MODULE_EXPORTS)

_LAZY_EXPORTS = {
    **{name: (_BENCHMARK_MODULE, name) for name in _BENCHMARK_EXPORTS},
    **{name: (_EVALUATION_MODULE, name) for name in _EVALUATION_EXPORTS},
    **{name: (_SYNTHESIS_MODULE, name) for name in _SYNTHESIS_EXPORTS},
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
