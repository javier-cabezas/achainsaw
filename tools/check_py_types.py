"""
Checks that the repository's Python code annotates its variables, which mypy --strict does
not require (it checks the annotations of functions, and that every type is consistent).

A name's first assignment in its scope (module, class or function) must carry its type,
`n: int = 0`, and so must a method's first assignment to `self.name` unless the class body
declares it. Unpacking a tuple display (`a, b = 0, 1`) is split into annotated assignments.
Names bound where Python has no place for an annotation take their types from what binds
them: loop and comprehension targets, unpacking a function's result, `with ... as`, `except
... as`, imports and `_`; a `self.name` bound this way is declared first, `self.name: int`.

Usage: python tools/check_py_types.py [paths...] (default: benchmarks, examples, tests, tools)
"""

import ast
import sys
from pathlib import Path
from typing import TypeAlias

DEFAULT_PATHS: list[str] = ["benchmarks", "examples", "tests", "tools"]

Scope: TypeAlias = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class Checker(ast.NodeVisitor):
    """Walks one file, reporting unannotated first assignments."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.errors: list[str] = []
        # Names already bound with a type in each enclosing scope, innermost last.
        self.scopes: list[set[str]] = []
        # Attributes declared or annotated per enclosing class, innermost last.
        self.attrs: list[set[str]] = []

    def report(self, node: ast.AST, what: str) -> None:
        line: int = getattr(node, "lineno", 0)
        self.errors.append(f"{self.path}:{line}: {what} needs a type annotation")

    def visit_scope(self, node: Scope, bound: set[str]) -> None:
        self.scopes.append(bound)
        for child in node.body:
            self.visit(child)
        self.scopes.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self.visit_scope(node, set())

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for d in node.decorator_list + node.bases:
            self.visit(d)
        declared: set[str] = {
            s.target.id
            for s in node.body
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
        }
        self.attrs.append(declared)
        self.visit_scope(node, set())
        self.attrs.pop()

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        a: ast.arguments = node.args
        params: list[ast.arg] = a.posonlyargs + a.args + a.kwonlyargs
        params += [p for p in (a.vararg, a.kwarg) if p is not None]
        # The parameters' annotations are mypy's to check (disallow_untyped_defs).
        self.visit_scope(node, {p.arg for p in params})

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        for a in node.names:
            self.scopes[-1].add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.visit_Import(node)

    def is_self_attr(self, t: ast.expr) -> bool:
        return (
            isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
            and bool(self.attrs)
        )

    def bind(self, t: ast.expr | None) -> None:
        """Binds the names in `t`, which takes its type from what binds it (a loop's
        iterable, a context manager)."""
        if t is not None:
            self.check_target(t, True)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.bind(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_withitem(self, node: ast.withitem) -> None:
        self.bind(node.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.scopes[-1].add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        t: ast.expr = node.target
        if isinstance(t, ast.Name):
            self.scopes[-1].add(t.id)
        elif isinstance(t, ast.Attribute) and self.is_self_attr(t):
            self.attrs[-1].add(t.attr)
        if node.value is not None:
            self.visit(node.value)

    def check_target(self, t: ast.expr, unpacked: bool) -> None:
        """Reports `t` if it binds a name or attribute for the first time; a name from
        unpacking a function's result (`unpacked`) takes its type from the function."""
        if isinstance(t, ast.Name):
            if t.id != "_" and t.id not in self.scopes[-1]:
                if not unpacked:
                    self.report(t, f"variable '{t.id}'")
                self.scopes[-1].add(t.id)
        elif isinstance(t, ast.Attribute) and self.is_self_attr(t):
            if t.attr not in self.attrs[-1]:
                self.report(t, f"attribute 'self.{t.attr}'")
                self.attrs[-1].add(t.attr)
        elif isinstance(t, ast.Tuple | ast.List):
            for e in t.elts:
                self.check_target(e, unpacked)
        elif isinstance(t, ast.Starred):
            self.check_target(t.value, unpacked)

    def visit_Assign(self, node: ast.Assign) -> None:
        display: bool = isinstance(node.value, ast.Tuple | ast.List)
        for t in node.targets:
            self.check_target(t, isinstance(t, ast.Tuple | ast.List) and not display)
        self.visit(node.value)


def python_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    return files


def main() -> int:
    errors: list[str] = []
    for f in python_files(sys.argv[1:] or DEFAULT_PATHS):
        checker: Checker = Checker(f)
        checker.visit(ast.parse(f.read_text(), str(f)))
        errors += checker.errors
    for e in errors:
        print(e)
    if errors:
        print(f"{len(errors)} unannotated variables")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
