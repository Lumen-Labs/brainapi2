"""Fail CI when new Cypher is assembled with string interpolation."""

from __future__ import annotations

import ast
import os
import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOTS = (_REPO_ROOT / "src", _REPO_ROOT / "deploy")
_ALLOWLIST = {
    _REPO_ROOT / "src" / "lib" / "cypher_safety.py",
}

_CYPHER_HINT = re.compile(
    r"\b(MATCH|MERGE|CREATE\s+DATABASE|DETACH\s+DELETE|OPTIONAL\s+MATCH|SHOW\s+DATABASE)\b",
    re.IGNORECASE,
)


def _constant_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
        return "".join(parts)
    return ""


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path in _ALLOWLIST:
                continue
            files.append(path)
    return files


def _find_interpolated_cypher(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[str] = []
    relative = path.relative_to(_REPO_ROOT)

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            has_interpolation = any(
                isinstance(value, ast.FormattedValue) for value in node.values
            )
            text = _constant_text(node)
            if has_interpolation and _CYPHER_HINT.search(text):
                hits.append(
                    f"{relative}:{node.lineno}: f-string interpolates Cypher"
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "format"
                and _CYPHER_HINT.search(_constant_text(func.value))
                and (node.args or node.keywords)
            ):
                hits.append(
                    f"{relative}:{node.lineno}: str.format interpolates Cypher"
                )
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"execute_query", "run"}
                and node.args
                and isinstance(node.args[0], ast.JoinedStr)
            ):
                hits.append(
                    f"{relative}:{node.lineno}: execute_query/run given an f-string"
                )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if _CYPHER_HINT.search(_constant_text(node.left)):
                hits.append(
                    f"{relative}:{node.lineno}: percent-format interpolates Cypher"
                )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if _is_quoted_value_concat(node):
                hits.append(
                    f"{relative}:{node.lineno}: concatenates a quoted Cypher value"
                )
    return hits


def _flatten_add(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_add(node.left) + _flatten_add(node.right)
    return [node]


def _is_quoted_value_concat(node: ast.BinOp) -> bool:
    parts = _flatten_add(node)
    texts = [_constant_text(part) for part in parts]
    joined = "".join(texts)
    if not _CYPHER_HINT.search(joined):
        return False
    for idx, text in enumerate(texts[:-1]):
        if text.endswith("'") or text.endswith('"'):
            nxt = parts[idx + 1]
            if not isinstance(nxt, ast.Constant):
                return True
    return False


class CypherInterpolationGuardTests(unittest.TestCase):
    def test_keyword_only_format_is_detected(self):
        tree = ast.parse(
            'query = "MATCH (n) WHERE n.name = \'{name}\' RETURN n".format(name=user_input)\n'
        )
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "format"
                and _CYPHER_HINT.search(_constant_text(func.value))
                and (node.args or node.keywords)
            ):
                hits.append("str.format interpolates Cypher")
        self.assertEqual(hits, ["str.format interpolates Cypher"])

    def test_no_interpolated_cypher_outside_helper(self):
        os.environ.setdefault("BRAINPAT_TOKEN", "test-token")
        violations = []
        for path in _iter_python_files():
            violations.extend(_find_interpolated_cypher(path))
        self.assertEqual(
            violations,
            [],
            "Cypher must be assembled with src.lib.cypher_safety helpers "
            "and query parameters, not f-strings/.format:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
