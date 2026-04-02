from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


DEFAULT_FILES = (
    "README.md",
    "README_cn.md",
    "README_ja.md",
    "README_ko.md",
    "docs/CLI_REFERENCE.md",
    "docs/START_HERE.md",
    "docs/STARTER_PATHS.md",
    "docs/BENCHMARK_GETTING_STARTED.md",
    "docs/README_DOCS.md",
    "docs/ALGORITHM_SELECTION_GUIDE.md",
    "docs/COMPARISON.md",
    "docs/source/index.rst",
    "docs/source/start_here.rst",
    "docs/source/starter_paths.rst",
    "docs/source/benchmark_getting_started.rst",
    "docs/source/examples.rst",
    "benchmarks/configs/README.md",
    "examples/README.md",
)

REQUIRED_NEEDLES: dict[str, tuple[str, ...]] = {
    "README.md": (
        "pyimgano-doctor --recommend-extras",
        "pyimgano-demo --smoke",
        "--list-starter-configs",
        "pyimgano --help",
    ),
    "README_cn.md": ("pyimgano-doctor", "pyimgano-demo --smoke", "--list-starter-configs"),
    "README_ja.md": ("pyimgano-doctor", "pyimgano-demo --smoke", "--list-starter-configs"),
    "README_ko.md": ("pyimgano-doctor", "pyimgano-demo --smoke", "--list-starter-configs"),
    "CLI_REFERENCE.md": (
        "--recommend-extras",
        "--for-command",
        "--for-model",
        "--smoke",
        "--summary-json",
        "--emit-next-steps",
        "--list-starter-configs",
        "--starter-config-info",
        "pyimgano --help",
    ),
    "START_HERE.md": ("pyimgano-doctor", "pyimgano-demo --smoke"),
    "STARTER_PATHS.md": (
        "pyimgano-doctor --profile first-run --json",
        "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json",
        "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json",
        "pyimgano --help",
    ),
    "BENCHMARK_GETTING_STARTED.md": ("--list-starter-configs", "--starter-config-info"),
    "README_DOCS.md": ("docs/START_HERE.md", "docs/STARTER_PATHS.md", "docs/BENCHMARK_GETTING_STARTED.md"),
    "ALGORITHM_SELECTION_GUIDE.md": ("If you are starting from the CLI", "pyim --list models --objective"),
    "COMPARISON.md": ("If You Are Deciding Where To Start", "pyimgano-demo --smoke"),
    "benchmarks/configs/README.md": ("--list-starter-configs", "--starter-config-info"),
    "examples/README.md": ("Goal", "Dependencies", "Offline-safe"),
    "index.rst": ("start_here", "benchmark_getting_started"),
    "start_here.rst": ("docs/START_HERE.md", "docs/BENCHMARK_GETTING_STARTED.md"),
    "starter_paths.rst": ("docs/STARTER_PATHS.md", "docs/START_HERE.md"),
    "benchmark_getting_started.rst": ("docs/BENCHMARK_GETTING_STARTED.md",),
    "examples.rst": ("examples/README.md",),
}


def _iter_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(str(raw))
        if path.is_file():
            yield path


def _required_needles_for(path: Path) -> tuple[str, ...]:
    relative_key = path.as_posix()
    if relative_key in REQUIRED_NEEDLES:
        return REQUIRED_NEEDLES[relative_key]
    key = path.name
    return REQUIRED_NEEDLES.get(key, ())


def _scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    for needle in _required_needles_for(path):
        if needle not in text:
            issues.append(f"{path}: missing required adoption doc entrypoint: {needle}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_adoption_docs",
        description="Fail when adoption-focused docs lose required entrypoints or starter workflow references.",
    )
    parser.add_argument("paths", nargs="*", help="Optional files to scan. Defaults to adoption docs.")
    args = parser.parse_args(argv)

    scan_paths = list(args.paths) if args.paths else list(DEFAULT_FILES)
    issues: list[str] = []
    for path in _iter_files(scan_paths):
        issues.extend(_scan(path))

    if issues:
        for issue in issues:
            print(issue)
        return 1

    print("OK: adoption docs include required entrypoints.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
