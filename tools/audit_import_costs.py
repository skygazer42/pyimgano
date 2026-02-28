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


_HEAVY_ROOT_MODULES = (
    # Common "keep out of import pyimgano" modules.
    "torch",
    "cv2",
    "open_clip",
    "diffusers",
    "faiss",
    "anomalib",
    "mamba_ssm",
)


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _heavy_roots_from_modules(modules: set[str]) -> set[str]:
    out: set[str] = set()
    for root in _HEAVY_ROOT_MODULES:
        if root in modules:
            out.add(root)
            continue
        prefix = f"{root}."
        if any(m.startswith(prefix) for m in modules):
            out.add(root)
    return out


def _time_import(mod: str) -> tuple[float, set[str]]:
    t0 = time.perf_counter()
    before = set(sys.modules.keys())
    importlib.import_module(mod)
    t1 = time.perf_counter()
    after = set(sys.modules.keys())

    newly_loaded = after - before
    heavy_new = _heavy_roots_from_modules(newly_loaded)
    return float(t1 - t0), heavy_new


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
    results: list[tuple[str, float, set[str]]] = []
    for m in mods:
        dt, heavy_new = _time_import(m)
        results.append((m, dt, heavy_new))

    results.sort(key=lambda kv: kv[1], reverse=True)
    print("Import costs (seconds):")
    for m, dt, heavy_new in results:
        extra = ""
        if heavy_new:
            extra = f"  (new heavy imports: {', '.join(sorted(heavy_new))})"
        print(f"- {m:30s} {dt:8.3f}s{extra}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
