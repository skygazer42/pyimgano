from __future__ import annotations

"""Audit helper for third-party code notices.

Policy:
- If we copy code into `pyimgano/`, we must keep upstream attribution.
- Copied files must contain the upstream marker.
- `third_party/NOTICE.md` must mention each referenced upstream repo.

This script is intentionally conservative and text-based.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpstreamRef:
    path: Path
    lineno: int
    marker: str


_GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
_ORG_REPO_RE = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b")
_UPSTREAM_MARKER = "UPSTREAM:"


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        out.append(p)
    return out


def _normalize_upstream(marker: str) -> tuple[str, str | None]:
    """Return (key, url) for notice matching and user-friendly reporting."""

    text = str(marker).strip()
    if not text:
        return "", None

    m = _GITHUB_RE.search(text)
    if m is not None:
        org_repo = m.group(1)
        return org_repo, f"https://github.com/{org_repo}"

    m = _ORG_REPO_RE.match(text)
    if m is not None:
        org_repo = m.group(1)
        return org_repo, f"https://github.com/{org_repo}"

    # Fallback: keep the raw marker as the key.
    return text, None


def _extract_upstreams(py_files: list[Path]) -> dict[str, list[UpstreamRef]]:
    upstream_to_refs: dict[str, list[UpstreamRef]] = {}
    for p in py_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if _UPSTREAM_MARKER not in text:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if _UPSTREAM_MARKER not in line:
                continue
            # Accept formats like:
            #   # UPSTREAM: org/repo @ <sha>
            #   # UPSTREAM: https://github.com/org/repo (commit ...)
            marker = line.split(_UPSTREAM_MARKER, 1)[1].strip()
            if not marker:
                continue
            key, _url = _normalize_upstream(marker)
            if not key:
                continue
            upstream_to_refs.setdefault(key, []).append(
                UpstreamRef(path=p, lineno=int(lineno), marker=marker)
            )
    return upstream_to_refs


def _report_missing_notice_file(
    *,
    repo_root: Path,
    upstreams: dict[str, list[UpstreamRef]],
) -> int:
    print("error: UPSTREAM markers found but third_party/NOTICE.md is missing", file=sys.stderr)
    for key, refs in sorted(upstreams.items(), key=lambda kv: kv[0].lower()):
        _key, url = _normalize_upstream(key)
        print(f"- {_UPSTREAM_MARKER} {key}", file=sys.stderr)
        if url is not None:
            print(f"  - url: {url}", file=sys.stderr)
        for ref in sorted(refs, key=lambda r: (str(r.path), int(r.lineno))):
            rel = ref.path.relative_to(repo_root)
            print(f"  - referenced by: {rel}:{ref.lineno}", file=sys.stderr)
    return 1


def _notice_needles(key: str, refs: list[UpstreamRef]) -> list[str]:
    normalized_key, url = _normalize_upstream(key)
    needles = [str(normalized_key)]
    if url is not None:
        needles.append(str(url))
    needles.extend([str(r.marker) for r in refs])
    return needles


def _collect_missing_notice_entries(
    *,
    upstreams: dict[str, list[UpstreamRef]],
    notice_l: str,
) -> dict[str, list[UpstreamRef]]:
    missing: dict[str, list[UpstreamRef]] = {}
    for key, refs in upstreams.items():
        if not any(needle.lower() in notice_l for needle in _notice_needles(key, refs) if needle):
            missing[key] = refs
    return missing


def _report_missing_notice_entries(
    *,
    repo_root: Path,
    missing: dict[str, list[UpstreamRef]],
    notice_l: str,
) -> int:
    print("error: missing third-party notice entries for copied code:", file=sys.stderr)
    for key, refs in sorted(missing.items(), key=lambda kv: kv[0].lower()):
        _key, url = _normalize_upstream(key)
        print(f"- UPSTREAM not found in third_party/NOTICE.md: {key}", file=sys.stderr)
        if url is not None and url.lower() not in notice_l:
            print(f"  - suggested url: {url}", file=sys.stderr)
        for ref in sorted(refs, key=lambda r: (str(r.path), int(r.lineno))):
            rel = ref.path.relative_to(repo_root)
            print(
                f"  - referenced by: {rel}:{ref.lineno}  ({_UPSTREAM_MARKER} {ref.marker})",
                file=sys.stderr,
            )
    print("", file=sys.stderr)
    print(
        "Fix: add an entry to third_party/NOTICE.md for each upstream repo referenced by UPSTREAM markers.",
        file=sys.stderr,
    )
    return 1


def main(_argv: list[str] | None = None) -> int:
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
        return _report_missing_notice_file(repo_root=repo_root, upstreams=upstreams)

    notice = notice_path.read_text(encoding="utf-8", errors="replace")
    notice_l = notice.lower()
    missing = _collect_missing_notice_entries(upstreams=upstreams, notice_l=notice_l)

    if missing:
        return _report_missing_notice_entries(
            repo_root=repo_root,
            missing=missing,
            notice_l=notice_l,
        )

    print(
        f"OK: found {len(upstreams)} UPSTREAM marker(s) and all are covered in third_party/NOTICE.md"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
