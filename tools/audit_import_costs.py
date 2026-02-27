from __future__ import annotations

"""Best-effort import cost audit.

This is a lightweight developer tool to catch accidental import-time explosions
(e.g. heavy downloads, huge model construction, noisy prints).

It is NOT a strict CI gate by default, but it is useful before releases.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _time_import(mod: str) -> float:
    t0 = time.perf_counter()
    importlib.import_module(mod)
    t1 = time.perf_counter()
    return float(t1 - t0)


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_root_on_sys_path()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modules",
        nargs="+",
        default=[
            "pyimgano",
            "pyimgano.models",
            "pyimgano.features",
            "pyimgano.synthesis",
        ],
        help="Modules to import and time",
    )
    args = parser.parse_args(argv)

    mods = [str(m) for m in args.modules]
    results: list[tuple[str, float]] = []
    for m in mods:
        dt = _time_import(m)
        results.append((m, dt))

    results.sort(key=lambda kv: kv[1], reverse=True)
    print("Import costs (seconds):")
    for m, dt in results:
        print(f"- {m:30s} {dt:8.3f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
