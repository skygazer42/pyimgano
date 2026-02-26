from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def audit_public_api() -> list[str]:
    """Return a list of human-readable issues with the public API."""

    _ensure_repo_root_on_sys_path()
    import pyimgano

    issues: list[str] = []
    names = list(getattr(pyimgano, "__all__", []))
    for name in names:
        try:
            getattr(pyimgano, name)
        except Exception as exc:  # noqa: BLE001 - tool boundary
            issues.append(f"{name}: {exc}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_public_api")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    issues = audit_public_api()
    ok = not issues

    if bool(args.json):
        payload: dict[str, Any] = {"ok": bool(ok), "issues": list(issues)}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if ok:
            print("OK: pyimgano public API looks consistent.")
        else:
            print("ERROR: pyimgano public API issues detected:", file=sys.stderr)
            for issue in issues:
                print(f"- {issue}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
