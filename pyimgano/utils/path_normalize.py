from __future__ import annotations

"""Cross-platform path normalization helpers.

Manifests (`.jsonl`) are often generated on one OS and consumed on another.
The most common portability issue is Windows-style separators (`\\`) being
treated as literal characters on POSIX.

These helpers normalize paths into a stable, forward-slash form suitable for:
- writing into manifests
- resolving manifest paths on POSIX/Windows

Notes
-----
- We do **not** call `.resolve()` here (it may require the file to exist and can
  change semantics).
- We keep URL-like strings (containing `://`) unchanged aside from trimming.
"""

from pathlib import Path


def normalize_path(value: str | Path) -> str:
    """Normalize filesystem paths for manifest IO (Windows + POSIX friendly)."""

    if isinstance(value, Path):
        text = value.as_posix()
    else:
        text = str(value)

    text = text.strip()
    if not text:
        return ""

    # Avoid corrupting URL-like references.
    if "://" in text:
        return text

    # Make separators stable across OSes.
    text = text.replace("\\", "/")

    # Best-effort: collapse accidental repeated separators, but preserve UNC-like
    # leading `//server/share` when present.
    if text.startswith("//"):
        prefix = "//"
        rest = text[2:]
    else:
        prefix = ""
        rest = text

    while "//" in rest:
        rest = rest.replace("//", "/")

    return prefix + rest

