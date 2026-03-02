from __future__ import annotations

import argparse
import sys
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

    issues: list[str] = []

    for name in sorted(list_models()):
        entry = MODEL_REGISTRY.info(name)
        tags = set(str(t) for t in entry.tags)
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
