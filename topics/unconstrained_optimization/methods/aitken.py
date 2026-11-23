"""Fixed-point sequences and Aitken estimates from the formula sheet."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from operator import index
from sys import float_info

import numpy as np
from numpy.typing import ArrayLike

MIN_POINT_COUNT = 3
DEFAULT_POINT_COUNT = 10


@dataclass(frozen=True)
class AitkenEstimate:
    """A Δ² estimate and its residual from three ordinary iterates."""

    index: int
    values: tuple[float, float, float]
    denominator: float
    estimate: float | None
    residual: float | None
    issue: str | None = None


@dataclass(frozen=True)
class FixedPointRun:
    points: tuple[float, ...]
    residuals: tuple[float | None, ...]
    estimates: tuple[AitkenEstimate, ...]
    stop_reason: str | None = None
    last_residual_issue: str | None = None


def iterate_fixed_point(
    fixed_point: Callable[[float], float], start: float, number_of_points: int = DEFAULT_POINT_COUNT
) -> FixedPointRun:
    """Produce the ordinary sequence, then estimate its limit without feedback."""

    number_of_points = index(number_of_points)
    if number_of_points < MIN_POINT_COUNT:
        raise ValueError(f"At least {MIN_POINT_COUNT} points are required.")
    start = float(start)
    if not isfinite(start):
        raise ValueError("The starting value must be finite.")

    points = [start]
    stop_reason = None
    for i in range(1, number_of_points):
        try:
            points.append(_evaluate_fixed_point(fixed_point, points[-1]))
        except (ArithmeticError, ValueError, TypeError) as error:
            stop_reason = f"Could not compute x[{i}]. {error}"
            break

    residuals: list[float | None] = [
        _absolute_difference(following, current) for current, following in zip(points, points[1:])
    ]
    last_residual = None
    residual_issue = None
    if stop_reason is None:
        try:
            # Probe g at the final point for its residual without adding another iterate.
            last_residual = _absolute_difference(_evaluate_fixed_point(fixed_point, points[-1]), points[-1])
            if last_residual is None:
                residual_issue = "The final ordinary residual overflowed."
        except (ArithmeticError, ValueError, TypeError) as error:
            residual_issue = f"The final ordinary residual is unavailable. {error}"
    residuals.append(last_residual)

    estimates = tuple(
        _estimate_fixed_point(fixed_point, i, (points[i - 2], points[i - 1], points[i]))
        for i in range(2, len(points))
    )
    return FixedPointRun(tuple(points), tuple(residuals), estimates, stop_reason, residual_issue)


def _estimate_fixed_point(
    fixed_point: Callable[[float], float], i: int, values: tuple[float, float, float]
) -> AitkenEstimate:
    older, previous, current = values
    delta, denominator = _aitken_differences(older, previous, current)
    if not isfinite(denominator):
        return AitkenEstimate(i, values, denominator, None, None, "The denominator D is not finite.")
    tolerance = (8 * float_info.epsilon) * max(map(abs, values))
    if abs(denominator) <= tolerance:
        return AitkenEstimate(i, values, denominator, None, None, "The denominator D is numerically zero.")

    estimate = _aitken_value(current, delta, denominator)
    if not isfinite(estimate):
        return AitkenEstimate(i, values, denominator, None, None, "The Aitken estimate is not finite.")
    try:
        residual = _absolute_difference(_evaluate_fixed_point(fixed_point, estimate), estimate)
    except (ArithmeticError, ValueError, TypeError) as error:
        return AitkenEstimate(i, values, denominator, estimate, None, f"Residual unavailable. {error}")
    issue = "The residual overflowed." if residual is None else None
    return AitkenEstimate(i, values, denominator, estimate, residual, issue)


def _evaluate_fixed_point(fixed_point: Callable[[float], float], point: float) -> float:
    try:
        value = fixed_point(point)
    except OverflowError as error:
        raise OverflowError("The evaluation of g(x) overflowed.") from error
    except RecursionError as error:
        raise ValueError("The expression for g(x) is too deeply nested.") from error
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError("g(x) must return a real scalar.")
    result = float(value)
    if not isfinite(result):
        raise ValueError("g(x) returned NaN or infinity.")
    return result


def _absolute_difference(left: float, right: float) -> float | None:
    residual = abs(left - right)
    return residual if isfinite(residual) else None


def _aitken_differences(older, previous, current):
    delta = current - previous
    return delta, delta - (previous - older)


def _aitken_value(current, delta, denominator):
    # Divide before multiplying to avoid squaring a large difference.
    return current - delta * (delta / denominator)


@dataclass(frozen=True)
class AitkenStep:
    """The two fixed-point iterates, accelerated point and fixed-point residual."""

    first: tuple[float, ...]
    second: tuple[float, ...]
    accelerated: tuple[float, ...]
    residual: float


def aitken_step(
    fixed_point: Callable[[np.ndarray], ArrayLike],
    point: ArrayLike,
    *,
    denominator_tolerance: float = 1e-12,
) -> AitkenStep:
    """Apply componentwise Aitken acceleration to G(x) and G(G(x))."""

    x0 = np.asarray(point, dtype=float)
    x1 = np.asarray(fixed_point(x0), dtype=float)
    x2 = np.asarray(fixed_point(x1), dtype=float)
    delta, denominator = _aitken_differences(x0, x1, x2)
    if np.any(~np.isfinite(denominator) | (np.abs(denominator) <= denominator_tolerance)):
        raise ZeroDivisionError("Aitken acceleration has a zero, small or non-finite denominator.")
    accelerated = _aitken_value(x2, delta, denominator)
    residual = float(np.linalg.norm(np.asarray(fixed_point(accelerated)) - accelerated))
    return AitkenStep(
        tuple(map(float, x1)), tuple(map(float, x2)), tuple(map(float, accelerated)), residual
    )
