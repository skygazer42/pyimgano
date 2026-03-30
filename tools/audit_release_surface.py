from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _load_pyproject_project_table() -> dict[str, object]:
    text = _read_text("pyproject.toml")
    project_started = False
    project_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            project_started = True
            continue
        if project_started and stripped.startswith("[") and stripped.endswith("]"):
            break
        if project_started:
            project_lines.append(line)

    project_text = "\n".join(project_lines)
    data: dict[str, object] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for raw_line in project_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if current_key is not None and current_list is not None:
            if line == "]":
                data[current_key] = current_list
                current_key = None
                current_list = None
                continue
            current_list.append(line.rstrip(",").strip().strip('"'))
            continue

        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value == "[":
            current_key = key
            current_list = []
            continue
        data[key] = value.strip('"')

    return data


def _run_help(module: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to run help for {module}: {proc.stderr}")
    return proc.stdout


def main() -> int:
    issues: list[str] = []

    project = _load_pyproject_project_table()
    if project.get("name") != "pyimgano":
        issues.append("pyproject [project].name must stay 'pyimgano'.")
    if project.get("readme") != "README.md":
        issues.append("pyproject [project].readme must stay 'README.md'.")

    readme = _read_text("README.md")
    cli_reference = _read_text("docs/CLI_REFERENCE.md")
    root_help = _run_help("pyimgano")
    doctor_help = _run_help("pyimgano.doctor_cli")
    pyim_help = _run_help("pyimgano.pyim_cli")

    required_pairs = [
        ("README.md", "pyimgano-doctor --recommend-extras"),
        ("README.md", "pyimgano-demo --smoke"),
        ("README.md", "--list-starter-configs"),
        ("README.md", "python -m pyimgano --help"),
        ("docs/CLI_REFERENCE.md", "--recommend-extras"),
        ("docs/CLI_REFERENCE.md", "--starter-config-info"),
        ("docs/CLI_REFERENCE.md", "python -m pyimgano --help"),
        ("python -m pyimgano --help", "pyimgano <command> [args...]"),
        ("python -m pyimgano --help", "benchmark --list-starter-configs"),
        ("python -m pyimgano.doctor_cli --help", "--recommend-extras"),
        ("python -m pyimgano.doctor_cli --help", "--for-command"),
        ("python -m pyimgano.doctor_cli --help", "--for-model"),
        ("python -m pyimgano.pyim_cli --help", "--objective"),
        ("python -m pyimgano.pyim_cli --help", "--selection-profile"),
        ("python -m pyimgano.pyim_cli --help", "--topk"),
    ]

    for source_name, needle in required_pairs:
        haystack = {
            "README.md": readme,
            "docs/CLI_REFERENCE.md": cli_reference,
            "python -m pyimgano --help": root_help,
            "python -m pyimgano.doctor_cli --help": doctor_help,
            "python -m pyimgano.pyim_cli --help": pyim_help,
        }[source_name]
        if needle not in haystack:
            issues.append(f"{source_name} missing required release-surface entry: {needle}")

    if issues:
        for issue in issues:
            print(issue)
        return 1

    print("OK: release surface is consistent across pyproject, docs, and root CLI help.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
