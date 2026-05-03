from __future__ import annotations

import ast
from pathlib import Path


def _uses_pep604_annotations(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False

        def _record_if_needed(self, annotation: ast.AST | None) -> None:
            if annotation is None:
                return
            if any(
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
                for node in ast.walk(annotation)
            ):
                self.found = True

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                self._record_if_needed(arg.annotation)
            if node.args.vararg is not None:
                self._record_if_needed(node.args.vararg.annotation)
            if node.args.kwarg is not None:
                self._record_if_needed(node.args.kwarg.annotation)
            self._record_if_needed(node.returns)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._record_if_needed(node.annotation)
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.found


def test_pep604_annotations_enable_postponed_evaluation_for_python39() -> None:
    offenders: list[str] = []

    for root in (Path("pyimgano"), Path("tests")):
        for path in sorted(root.rglob("*.py")):
            if not _uses_pep604_annotations(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            body = list(tree.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]

            if not body:
                offenders.append(path.as_posix())
                continue

            first_stmt = body[0]
            if not (
                isinstance(first_stmt, ast.ImportFrom)
                and first_stmt.module == "__future__"
                and any(alias.name == "annotations" for alias in first_stmt.names)
            ):
                offenders.append(path.as_posix())

    assert offenders == []
