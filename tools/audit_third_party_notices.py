from __future__ import annotations

"""Audit helper for third-party code notices.

Policy:
- If we copy code into `pyimgano/`, we must keep upstream attribution.
- Copied files must contain an `UPSTREAM:` marker.
- `third_party/NOTICE.md` must mention each upstream repo referenced by `UPSTREAM:`.

This script is intentionally conservative and text-based.
"""

import sys
from pathlib import Path


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if p.name == "__pycache__":
            continue
        out.append(p)
    return out


def _extract_upstreams(py_files: list[Path]) -> dict[str, list[Path]]:
    upstream_to_files: dict[str, list[Path]] = {}
    for p in py_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "UPSTREAM:" not in text:
            continue

        for line in text.splitlines():
            if "UPSTREAM:" not in line:
                continue
            # Accept formats like:
            #   # UPSTREAM: org/repo @ <sha>
            #   # UPSTREAM: https://github.com/org/repo (commit ...)
            marker = line.split("UPSTREAM:", 1)[1].strip()
            if not marker:
                continue
            upstream = marker
            upstream_to_files.setdefault(upstream, []).append(p)
    return upstream_to_files


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "pyimgano"
    notice_path = repo_root / "third_party" / "NOTICE.md"

    if not src_root.exists():
        print(f"error: expected source root at {src_root}", file=sys.stderr)
        return 2

    py_files = _iter_py_files(src_root)
    upstreams = _extract_upstreams(py_files)

    if not upstreams:
        # Nothing to audit.
        print("OK: no UPSTREAM markers found in pyimgano/ (no copied code detected).")
        return 0

    if not notice_path.exists():
        print("error: UPSTREAM markers found but third_party/NOTICE.md is missing", file=sys.stderr)
        for up, files in upstreams.items():
            print(f"- UPSTREAM: {up}", file=sys.stderr)
            for f in sorted(files):
                rel = f.relative_to(repo_root)
                print(f"  - file: {rel}", file=sys.stderr)
        return 1

    notice = notice_path.read_text(encoding="utf-8", errors="replace")

    missing: dict[str, list[Path]] = {}
    for up, files in upstreams.items():
        # Require that the upstream marker text appears somewhere in NOTICE.md.
        # This is strict but keeps compliance easy to reason about.
        if up not in notice:
            missing[up] = files

    if missing:
        print("error: missing third-party notice entries for copied code:", file=sys.stderr)
        for up, files in sorted(missing.items(), key=lambda kv: kv[0].lower()):
            print(f"- UPSTREAM not found in third_party/NOTICE.md: {up}", file=sys.stderr)
            for f in sorted(files):
                rel = f.relative_to(repo_root)
                print(f"  - referenced by: {rel}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix: add an entry to third_party/NOTICE.md for each UPSTREAM marker.", file=sys.stderr)
        return 1

    print(f"OK: found {len(upstreams)} UPSTREAM marker(s) and all are covered in third_party/NOTICE.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

