from __future__ import annotations

"""Audit: forbid TensorRT (`tensorrt` / `trt`) imports in the codebase.

Industrial rationale
--------------------
- TensorRT is an accelerator backend that is often unavailable in CI and many
  production environments.
- Importing it at module import time can crash the process (missing shared libs),
  and violates our "offline-safe / lightweight import" expectations.

Policy
------
- Do not import `tensorrt` (directly or via `import tensorrt as trt`) anywhere in
  `pyimgano/`.
- Do not import a `trt` module alias directly.

If TensorRT support is ever added in the future, it must remain fully optional,
use lazy imports behind explicit feature flags, and must not be imported during
package/module import.
"""

import re
import sys
from pathlib import Path

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("import tensorrt", re.compile(r"^\s*import\s+tensorrt\b", re.MULTILINE)),
    ("from tensorrt import", re.compile(r"^\s*from\s+tensorrt\b\s+import\b", re.MULTILINE)),
    ("import tensorrt as trt", re.compile(r"^\s*import\s+tensorrt\s+as\s+trt\b", re.MULTILINE)),
    ("import trt", re.compile(r"^\s*import\s+trt\b", re.MULTILINE)),
    ("from trt import", re.compile(r"^\s*from\s+trt\b\s+import\b", re.MULTILINE)),
]


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _line_for_offset(text: str, offset: int) -> int:
    if offset <= 0:
        return 1
    return 1 + text.count("\n", 0, offset)


def audit_no_tensorrt_imports(*, root: Path) -> list[str]:
    errors: list[str] = []
    for path in _iter_py_files(root):
        try:
            txt = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 - audit helper
            # Ignore unreadable files (should not happen in normal repos).
            continue

        for label, pat in _PATTERNS:
            m = pat.search(txt)
            if m is None:
                continue
            line = _line_for_offset(txt, m.start())
            errors.append(f"{path}:{line}: forbidden TensorRT import ({label})")

    return errors


def main(argv: list[str] | None = None) -> int:
    _ = argv
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "pyimgano"

    errors = audit_no_tensorrt_imports(root=pkg_root)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print("FAIL: TensorRT imports are forbidden in pyimgano/.", file=sys.stderr)
        return 1

    print("OK: no TensorRT (`tensorrt` / `trt`) imports found in pyimgano/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
