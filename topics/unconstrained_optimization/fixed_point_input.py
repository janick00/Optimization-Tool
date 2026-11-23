"""Read a real scalar iteration function from a mathematical expression."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable


_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: math.pow,
}
_FUNCTIONS = {
    name: getattr(math, name)
    for name in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                 "exp", "log", "log10", "sqrt", "floor", "ceil")
}
_FUNCTIONS.update({"ln": math.log, "abs": abs, "min": min, "max": max})
_CONSTANTS = {"pi": math.pi, "e": math.e, "E": math.e}


def parse_fixed_point_function(expression: str) -> Callable[[float], float]:
    """Accept scalar arithmetic in x and common real mathematical functions."""

    try:
        tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")
        return _compile(tree.body)
    except (SyntaxError, OverflowError, RecursionError) as error:
        raise ValueError("Enter a mathematical function of x, for example cos(x).") from error


def _compile(node: ast.AST) -> Callable[[float], float]:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)
        if not math.isfinite(value):
            raise ValueError("Numeric constants must be finite.")
        return lambda x: value
    if isinstance(node, ast.Name):
        if node.id == "x":
            return lambda x: x
        if node.id in _CONSTANTS:
            return lambda x: _CONSTANTS[node.id]
        raise ValueError(f"Unknown name {node.id!r}. Use x as the iteration variable.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _compile(node.operand)
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return lambda x: sign * operand(x)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        operation = _BINARY[type(node.op)]
        left, right = _compile(node.left), _compile(node.right)
        return lambda x: operation(left(x), right(x))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError(f"Unknown mathematical function {node.func.id!r}.")
        arguments = tuple(map(_compile, node.args))
        return lambda x: function(*(argument(x) for argument in arguments))
    raise ValueError("Use real numbers, x, arithmetic and mathematical functions such as cos(x).")
