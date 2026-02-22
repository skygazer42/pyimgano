#!/usr/bin/env python3
"""Generate a markdown index of registered models.

This script is intentionally **static**: it does not import `pyimgano` (or any
third-party deps). Instead it parses Python source via `ast` and extracts
`@register_model(...)` decorators.

Usage
-----
    python tools/generate_model_index.py

Outputs
-------
    docs/MODEL_INDEX.md
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "pyimgano" / "models"
OUT_PATH = ROOT / "docs" / "MODEL_INDEX.md"


@dataclass(frozen=True)
class Registration:
    name: str
    module: str
    target: str
    tags: tuple[str, ...]
    year: Optional[int]
    description: Optional[str]
    backend: Optional[str]


def _is_register_model_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "register_model"
    if isinstance(func, ast.Attribute):
        return func.attr == "register_model"
    return False


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _extract_str_seq(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in node.elts:
            value = _const_str(elt)
            if value is None:
                return ()
            out.append(value)
        return tuple(out)
    return ()


def _extract_metadata(node: ast.AST) -> dict[str, Any]:
    if not isinstance(node, ast.Dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in zip(node.keys, node.values):
        key = _const_str(k) if k is not None else None
        if key is None:
            continue
        if isinstance(v, ast.Constant):
            out[key] = v.value
        else:
            # Non-constant metadata (e.g. function calls) is ignored on purpose.
            continue
    return out


def _extract_registration_from_decorators(
    decorators: Iterable[ast.expr],
    *,
    module: str,
    target: str,
) -> list[Registration]:
    regs: list[Registration] = []
    for deco in decorators:
        if not _is_register_model_call(deco):
            continue
        call = deco
        if not call.args:
            continue
        name = _const_str(call.args[0])
        if name is None:
            continue

        tags: tuple[str, ...] = ()
        metadata: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg == "tags" and kw.value is not None:
                tags = _extract_str_seq(kw.value)
            if kw.arg == "metadata" and kw.value is not None:
                metadata = _extract_metadata(kw.value)

        year_val = metadata.get("year", None)
        year = int(year_val) if isinstance(year_val, int) else None
        description_val = metadata.get("description", None)
        description = str(description_val) if isinstance(description_val, str) else None
        backend_val = metadata.get("backend", None)
        backend = str(backend_val) if isinstance(backend_val, str) else None

        regs.append(
            Registration(
                name=name,
                module=module,
                target=target,
                tags=tags,
                year=year,
                description=description,
                backend=backend,
            )
        )
    return regs


def _extract_registrations_from_file(path: Path) -> list[Registration]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    module = path.relative_to(ROOT).as_posix()
    regs: list[Registration] = []

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            regs.extend(
                _extract_registration_from_decorators(
                    node.decorator_list,
                    module=module,
                    target=node.name,
                )
            )
    return regs


def _render_markdown(regs: list[Registration]) -> str:
    regs_sorted = sorted(regs, key=lambda r: r.name)

    lines: list[str] = []
    lines.append("# Model Index")
    lines.append("")
    lines.append(
        "This file is auto-generated from `pyimgano/models/*` by "
        "`tools/generate_model_index.py`."
    )
    lines.append("")
    lines.append(f"Total registered model names: **{len(regs_sorted)}**")
    lines.append("")
    lines.append("| Name | Tags | Year | Backend | Description | Module |")
    lines.append("|---|---|---:|---|---|---|")
    for r in regs_sorted:
        tags = ", ".join(r.tags) if r.tags else ""
        year = str(r.year) if r.year is not None else ""
        backend = r.backend or ""
        desc = (r.description or "").replace("\n", " ").strip()
        module = f"`{r.module}`"
        lines.append(f"| `{r.name}` | {tags} | {year} | {backend} | {desc} | {module} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not MODELS_DIR.is_dir():
        raise RuntimeError(f"Models dir not found: {MODELS_DIR}")

    regs: list[Registration] = []
    for path in sorted(MODELS_DIR.glob("*.py")):
        if path.name in {"__init__.py", "registry.py"}:
            continue
        regs.extend(_extract_registrations_from_file(path))

    # Dedupe by name (keep the last occurrence so later overrides win).
    deduped: dict[str, Registration] = {}
    for r in regs:
        deduped[r.name] = r

    out = _render_markdown(list(deduped.values()))
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({len(deduped)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
