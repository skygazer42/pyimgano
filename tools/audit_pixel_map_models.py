from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def audit_pixel_map_models() -> list[str]:
    """Validate that pixel-map models expose anomaly-map methods.

    Rules (best-effort):
    - If a model has tag `pixel_map`, its constructor must define at least one
      of: `predict_anomaly_map`, `get_anomaly_map`.
    - If a constructor defines a map method but is missing the tag, warn.
    """

    _ensure_repo_root_on_sys_path()
    import pyimgano.models  # noqa: F401 - populate registry
    from pyimgano.models.registry import MODEL_REGISTRY, list_models
    from pyimgano.utils.extras import extra_for_root_module

    issues: list[str] = []
    imported_modules: set[str] = set()

    for name in sorted(list_models()):
        entry = MODEL_REGISTRY.info(name)
        tags = {str(t) for t in entry.tags}
        ctor = entry.constructor

        # Lazy registry entries are placeholders. For pixel-map contract auditing,
        # materialize the real constructor by importing the owning module.
        if "pixel_map" in tags and bool(entry.metadata.get("_lazy_placeholder", False)):
            module_name = str(entry.metadata.get("_lazy_module", "")).strip()
            if module_name and module_name not in imported_modules:
                try:
                    import_module(f"pyimgano.models.{module_name}")
                except ModuleNotFoundError as exc:
                    # Optional pixel-map models (deep/torch backends) may not be importable
                    # in a minimal environment. Treat missing optional deps as a skip,
                    # not a contract issue.
                    missing = getattr(exc, "name", None)
                    root = (str(missing).split(".", 1)[0] if missing else "").strip()
                    if root and extra_for_root_module(root) is not None:
                        continue
                    issues.append(f"{name}: failed to import module {module_name!r}: {exc}")
                    continue
                except ImportError as exc:
                    # Many optional-deps gates raise ImportError with a `pyimgano[...]` hint.
                    # Skip those modules; they can't be audited without the extra installed.
                    if "pyimgano[" in str(exc):
                        continue
                    issues.append(f"{name}: failed to import module {module_name!r}: {exc}")
                    continue
                except Exception as exc:  # noqa: BLE001 - tool boundary
                    issues.append(f"{name}: failed to import module {module_name!r}: {exc}")
                    continue
                imported_modules.add(module_name)

            entry = MODEL_REGISTRY.info(name)
            tags = {str(t) for t in entry.tags}
            ctor = entry.constructor

        has_map_method = bool(
            hasattr(ctor, "predict_anomaly_map") or hasattr(ctor, "get_anomaly_map")
        )

        if "pixel_map" in tags and not has_map_method:
            issues.append(
                f"{name}: tagged pixel_map but constructor has no predict_anomaly_map/get_anomaly_map"
            )
        if has_map_method and "pixel_map" not in tags:
            issues.append(f"{name}: defines anomaly-map method(s) but is missing pixel_map tag")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_pixel_map_models")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any issues are found (default: warn only).",
    )
    args = parser.parse_args(argv)

    issues = audit_pixel_map_models()
    if issues:
        print("WARN: pixel-map audit issues detected:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1 if bool(getattr(args, "strict", False)) else 0

    print("OK: pixel-map model tags match anomaly-map method availability.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
