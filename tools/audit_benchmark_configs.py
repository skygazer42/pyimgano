from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyimgano.reporting.benchmark_config import load_and_validate_benchmark_config


DEFAULT_DOC_PATHS = (
    "README.md",
    "docs",
    "benchmarks/configs/README.md",
)

_CONFIG_REF_PATTERN = re.compile(r"\b(official_[a-z0-9_]+\.json)\b")


def _iter_doc_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.suffix.lower() not in {".md", ".rst", ".txt"}:
                continue
            out.append(candidate)
    return out


def _scan_doc_config_refs(paths: list[str], known_names: set[str]) -> list[str]:
    issues: list[str] = []
    for path in _iter_doc_files(paths):
        text = path.read_text(encoding="utf-8")
        for match in _CONFIG_REF_PATTERN.finditer(text):
            name = str(match.group(1))
            if name not in known_names:
                issues.append(f"{path}: unknown benchmark config reference: {name}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_benchmark_configs",
        description="Validate official benchmark configs and referenced config names in docs.",
    )
    parser.add_argument(
        "--docs",
        nargs="*",
        help="Optional files/directories to scan for referenced official config names.",
    )
    args = parser.parse_args(argv)

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

    doc_issues = _scan_doc_config_refs(
        list(args.docs) if args.docs else list(DEFAULT_DOC_PATHS),
        {path.name for path in paths},
    )
    for issue in doc_issues:
        print(issue)
        failed = True
    if failed:
        return 1

    print(f"OK: validated {len(paths)} official benchmark config(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
