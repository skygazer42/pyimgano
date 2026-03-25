from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def audit_registry(*, limit: int | None = None) -> list[str]:
    """Validate that registry entries can be introspected and JSON-encoded."""

    _ensure_repo_root_on_sys_path()
    import pyimgano.models as models
    from pyimgano.models.registry import model_info
    from pyimgano.utils.extras import extra_for_root_module
    from pyimgano.utils.jsonable import to_jsonable

    issues: list[str] = []
    names = models.list_models()
    if limit is not None:
        names = names[: int(limit)]

    for name in names:
        try:
            info = model_info(name)
            json.dumps(to_jsonable(info))
        except ModuleNotFoundError as exc:
            root = _optional_missing_root(exc)
            if root is not None and extra_for_root_module(root) is not None:
                continue
            issues.append(f"{name}: {exc}")
        except ImportError as exc:
            root = _optional_missing_root(exc)
            if root is not None and extra_for_root_module(root) is not None:
                continue
            issues.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - tool boundary
            issues.append(f"{name}: {exc}")
    return issues


def _optional_missing_root(exc: ImportError) -> str | None:
    name = str(getattr(exc, "name", "") or "").strip()
    if name:
        return name.split(".", 1)[0]

    message = str(exc)
    quoted = re.findall(r"'([^']+)'", message)
    for candidate in quoted:
        root = str(candidate).strip().split(".", 1)[0]
        if root:
            return root

    no_module = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
    if no_module is not None:
        return str(no_module.group(1)).split(".", 1)[0]

    return None


def audit_registry_metadata(*, limit: int | None = None) -> dict[str, Any]:
    """Audit registry metadata against the structured metadata contract."""

    _ensure_repo_root_on_sys_path()
    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import audit_model_metadata

    return audit_model_metadata(limit=limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_registry")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional limit for models to check"
    )
    parser.add_argument(
        "--metadata-contract",
        action="store_true",
        help="Audit registry metadata contract instead of JSON-encoding model_info payloads.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    if bool(args.metadata_contract):
        payload = audit_registry_metadata(limit=args.limit)
        summary = payload["summary"]
        ok = not (
            summary["models_with_required_issues"]
            or summary["models_with_recommended_issues"]
            or summary["models_with_invalid_fields"]
        )
        if bool(args.json):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                "Metadata audit: "
                f"required={summary['models_with_required_issues']} "
                f"recommended={summary['models_with_recommended_issues']} "
                f"invalid={summary['models_with_invalid_fields']}"
            )
        return 0 if ok else 1

    issues = audit_registry(limit=args.limit)
    ok = not issues

    if bool(args.json):
        payload: dict[str, Any] = {"ok": bool(ok), "issues": list(issues)}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if ok:
            print("OK: registry model_info payloads are JSON-friendly.")
        else:
            print("ERROR: registry issues detected:", file=sys.stderr)
            for issue in issues:
                print(f"- {issue}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
