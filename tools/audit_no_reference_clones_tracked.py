from __future__ import annotations

"""Audit: ensure study-only reference clones are not tracked by git.

We use `.cache/pyimgano_refs/` as the default destination for shallow study clones
via `tools/clone_reference_repos.sh`. Those clones must never be committed.
"""

import subprocess
import sys
from pathlib import Path


def _git_ls_files(repo_root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to run this audit") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed:\n{proc.stderr}".rstrip())

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001 - conventional signature
    repo_root = Path(__file__).resolve().parents[1]
    tracked = _git_ls_files(repo_root)

    bad_prefix = ".cache/pyimgano_refs/"
    bad = sorted([p for p in tracked if p == bad_prefix.rstrip("/") or p.startswith(bad_prefix)])
    if bad:
        print("error: study-only reference clones are tracked by git:", file=sys.stderr)
        for p in bad[:50]:
            print(f"- {p}", file=sys.stderr)
        if len(bad) > 50:
            print(f"... ({len(bad) - 50} more)", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix:", file=sys.stderr)
        print("- Remove them from the git index (keep files on disk):", file=sys.stderr)
        print("  git rm -r --cached .cache/pyimgano_refs", file=sys.stderr)
        print("- Ensure `.cache/pyimgano_refs/` is ignored by git.", file=sys.stderr)
        return 1

    print("OK: no tracked files under .cache/pyimgano_refs/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
