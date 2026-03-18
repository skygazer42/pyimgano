from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyimgano.reporting.benchmark_config import load_and_validate_benchmark_config


def main() -> int:
    cfg_dir = Path("benchmarks/configs")
    paths = sorted(cfg_dir.glob("official_*.json"))
    if not paths:
        print("error: no official benchmark configs found", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        try:
            load_and_validate_benchmark_config(path)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {path}: {exc}", file=sys.stderr)
            failed = True
    if failed:
        return 1

    print(f"OK: validated {len(paths)} official benchmark config(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
