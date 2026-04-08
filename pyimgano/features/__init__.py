"""Feature extractor utilities for classical detectors."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Sequence

from .base import BaseFeatureExtractor
from .protocols import FeatureExtractor, FittableFeatureExtractor
from .registry import (
    FEATURE_REGISTRY,
    create_feature_extractor,
    feature_info,
    list_feature_extractors,
    materialize_feature_extractor,
    register_feature_extractor,
)

__all__ = [
    "BaseFeatureExtractor",
    "IdentityExtractor",
    "FeatureExtractor",
    "FittableFeatureExtractor",
    "FEATURE_REGISTRY",
    "create_feature_extractor",
    "feature_info",
    "list_feature_extractors",
    "materialize_feature_extractor",
    "register_feature_extractor",
]


_FEATURE_MODULE_ALLOWLIST: tuple[str, ...] = (
    "identity",
    "hog",
    "lbp",
    "gabor",
    "color_hist",
    "edge_stats",
    "fft_lowfreq",
    "patch_stats",
    "structural",
    "torchvision_backbone",
    "torchvision_backbone_gem",
    "torchvision_multilayer",
    "torchvision_vit_tokens",
    "torchvision_patch_tokens",
    "patch_grid",
    "torchscript_embed",
    "onnx_embed",
    "openclip_embed",
    "multi",
    "pca_projector",
    "scaler",
    "normalize",
)


def _module_source_path(module_name: str) -> Optional[Path]:
    pkg_root = Path(__file__).resolve().parent
    mod_path = pkg_root / f"{module_name}.py"
    if mod_path.is_file():
        return mod_path
    pkg_init = pkg_root / module_name / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _as_register_feature_extractor_call(node: ast.AST) -> Optional[ast.Call]:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "register_feature_extractor":
        return node
    if isinstance(func, ast.Attribute) and func.attr == "register_feature_extractor":
        return node
    return None


def _extract_feature_name(call: ast.Call) -> Optional[str]:
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return str(first.value)
    for kw in call.keywords:
        if (
            kw.arg == "name"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return str(kw.value.value)
    return None


def _extract_tags(call: ast.Call) -> tuple[str, ...]:
    for kw in call.keywords:
        if kw.arg != "tags":
            continue
        value = _safe_literal_eval(kw.value)
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(str(v) for v in value))
    return ()


def _extract_metadata(call: ast.Call) -> dict[str, Any]:
    for kw in call.keywords:
        if kw.arg != "metadata":
            continue
        value = _safe_literal_eval(kw.value)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _scan_module_for_extractors(
    module_name: str,
) -> list[tuple[str, tuple[str, ...], dict[str, Any]]]:
    path = _module_source_path(module_name)
    if path is None:
        return []

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    out: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for dec in getattr(node, "decorator_list", ()):
            call = _as_register_feature_extractor_call(dec)
            if call is None:
                continue

            feature_name = _extract_feature_name(call)
            if feature_name is None:
                continue

            out.append((feature_name, _extract_tags(call), _extract_metadata(call)))
    return out


def _make_lazy_feature_constructor(*, feature_name: str, module_name: str):
    def _ctor(*args: Any, **kwargs: Any):  # noqa: ANN401 - generic factory signature
        try:
            import_module(f"{__name__}.{module_name}")
        except ModuleNotFoundError as exc:
            from pyimgano.utils.extras import extra_for_root_module, extras_install_hint

            missing = getattr(exc, "name", None)
            root = (str(missing).split(".", 1)[0] if missing else "").strip()
            extra = extra_for_root_module(root)
            if extra is not None:
                raise ImportError(
                    f"Optional dependency {root!r} is required to construct feature extractor "
                    f"{feature_name!r} (implementation module {module_name!r}).\n"
                    "Install it via:\n"
                    f"  {extras_install_hint([extra])}\n"
                    f"Original error: {exc}"
                ) from exc
            raise

        real = FEATURE_REGISTRY.get(feature_name)
        if real is _ctor:
            raise RuntimeError(
                "Lazy feature extractor "
                f"{feature_name!r} failed to resolve after importing module {module_name!r}."
            )
        return real(*args, **kwargs)

    _ctor.__name__ = f"lazy_{feature_name}"
    _ctor.__qualname__ = f"lazy_{feature_name}"
    _ctor.__module__ = __name__
    setattr(_ctor, "_pyimgano_lazy", {"name": feature_name, "module": module_name})
    return _ctor


def _register_lazy_extractors(modules: Sequence[str]) -> None:
    for module_name in modules:
        for feature_name, tags, metadata in _scan_module_for_extractors(str(module_name)):
            meta = dict(metadata)
            meta["_lazy_placeholder"] = True
            meta["_lazy_module"] = str(module_name)
            FEATURE_REGISTRY.register(
                str(feature_name),
                _make_lazy_feature_constructor(
                    feature_name=str(feature_name), module_name=str(module_name)
                ),
                tags=tags,
                metadata=meta,
                overwrite=False,
            )


_register_lazy_extractors(_FEATURE_MODULE_ALLOWLIST)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "IdentityExtractor": ("identity", "IdentityExtractor"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy export
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - tooling convenience
    return sorted(set(globals()) | set(__all__))
