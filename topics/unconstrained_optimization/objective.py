"""Parse a symbolic objective and evaluate it and its derivatives with NumPy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class ParsedFunction:
    """SymPy representation and numerical derivatives of one function."""

    expression: sp.Expr
    symbols: tuple[sp.Symbol, ...]
    gradient_expression: tuple[sp.Expr, ...]
    hessian_expression: sp.Matrix
    _function: Callable[..., object]
    _gradient_function: Callable[..., object]
    _hessian_function: Callable[..., object]

    @property
    def dimension(self) -> int:
        return len(self.symbols)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(str(symbol) for symbol in self.symbols)

    def evaluate(self, point: Sequence[float]) -> float:
        """Evaluate f at a numerical point."""

        values = validate_point(point, self.dimension)
        result = np.asarray(self._function(*values), dtype=float).reshape(-1)
        if result.size != 1 or not np.isfinite(result[0]):
            raise ValueError(f"Function is not finite at point {_format_point(values)}.")
        return float(result[0])

    def evaluate_grid(self, *coordinates: np.ndarray | float) -> np.ndarray:
        """Evaluate f on broadcastable coordinates, using NaN outside its domain."""

        if len(coordinates) != self.dimension:
            raise ValueError(f"Grid must contain exactly {self.dimension} coordinates.")
        arrays = (np.asarray(coordinate, dtype=float) for coordinate in coordinates)
        grids = np.broadcast_arrays(*arrays)
        shape = grids[0].shape
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            try:
                values = np.asarray(self._function(*grids), dtype=float)
                values = np.broadcast_to(values, shape)
            except (ValueError, TypeError, OverflowError, FloatingPointError):
                # Some NumPy lambdify expressions can only be evaluated point by point.
                return self._evaluate_grid_pointwise(grids)
        return np.where(np.isfinite(values), values, np.nan)

    def _evaluate_grid_pointwise(self, grids: Sequence[np.ndarray]) -> np.ndarray:
        values = np.empty(grids[0].size, dtype=float)
        for index, point in enumerate(zip(*(grid.flat for grid in grids))):
            try:
                values[index] = self.evaluate(point)
            except (ValueError, OverflowError, FloatingPointError):
                values[index] = np.nan
        return values.reshape(grids[0].shape)

    def gradient_at(self, point: Sequence[float]) -> np.ndarray:
        """Evaluate the analytic gradient at a numerical point."""

        values = validate_point(point, self.dimension)
        result = np.asarray(self._gradient_function(*values), dtype=float).reshape(-1)
        if result.size != self.dimension or not np.all(np.isfinite(result)):
            raise ValueError(f"Gradient is not finite at point {_format_point(values)}.")
        return result

    def hessian_at(self, point: Sequence[float]) -> np.ndarray:
        """Evaluate the analytic Hessian at a numerical point."""

        values = validate_point(point, self.dimension)
        result = np.asarray(self._hessian_function(*values), dtype=float)
        result = result.reshape((self.dimension, self.dimension))
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Hessian is not finite at point {_format_point(values)}.")
        return result


def parse_function(expression_text: str, variable_text: str) -> ParsedFunction:
    """Parse a user expression and create analytic first/second derivatives."""

    variable_names = tuple(
        token for token in re.split(r"[,;\s]+", variable_text.strip()) if token
    )
    if not variable_names:
        raise ValueError("Enter at least one variable, for example x1, x2.")
    if any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in variable_names):
        raise ValueError("Variable names may contain letters, numbers, and underscores only.")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Variable names must be unique.")
    if not expression_text.strip():
        raise ValueError("Enter a function expression.")

    symbols = tuple(sp.Symbol(name, real=True) for name in variable_names)
    local_symbols = dict(zip(variable_names, symbols))
    try:
        expression = sp.sympify(expression_text.replace("^", "**"), locals=local_symbols)
    except (sp.SympifyError, TypeError, ValueError) as error:
        raise ValueError(f"Could not parse the function: {error}") from error

    unknown_symbols = expression.free_symbols.difference(symbols)
    if unknown_symbols:
        unknown = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        raise ValueError(f"Unknown symbols in the function: {unknown}.")
    if expression.is_real is False:
        raise ValueError("The function must be real-valued.")

    gradient = tuple(sp.diff(expression, symbol) for symbol in symbols)
    hessian = sp.hessian(expression, symbols)
    try:
        function = sp.lambdify(symbols, expression, modules="numpy")
        gradient_function = sp.lambdify(symbols, gradient, modules="numpy")
        hessian_function = sp.lambdify(symbols, hessian, modules="numpy")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Could not create numerical derivatives: {error}") from error

    return ParsedFunction(
        expression=expression,
        symbols=symbols,
        gradient_expression=gradient,
        hessian_expression=hessian,
        _function=function,
        _gradient_function=gradient_function,
        _hessian_function=hessian_function,
    )


def validate_point(point: Sequence[float], dimension: int) -> np.ndarray:
    """Convert coordinates to a finite vector of the expected length."""

    values = np.asarray(tuple(float(value) for value in point), dtype=float)
    if values.size != dimension:
        raise ValueError(f"Start point must contain exactly {dimension} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Start point must contain finite numbers.")
    return values


def _format_point(point: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):.8g}" for value in point) + ")"
