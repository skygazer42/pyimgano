from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "pyimgano"


def _iter_root_service_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyimgano.services":
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pyimgano.services":
                imported = ", ".join(alias.name for alias in node.names)
                violations.append(f"{node.module}: {imported}")

    return violations


def test_source_modules_do_not_import_from_service_root_package() -> None:
    violations: list[str] = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT)
        if rel.as_posix() == "services/__init__.py":
            continue
        for imported in _iter_root_service_imports(path):
            violations.append(f"{rel.as_posix()}: {imported}")

    assert violations == []
