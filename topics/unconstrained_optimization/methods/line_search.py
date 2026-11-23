"""Successive halving and parabola fitting for gradient descent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_MAX_HALVINGS = 50
DEFAULT_PARABOLA_BETA = 0.25


@dataclass(frozen=True)
class LineSearchTrial:
    """Objective value at a trial point and its comparison with the reference."""

    step_size: float
    point: tuple[float, ...]
    value: float | None
    accepted: bool
    phase: str
    rejection_reason: str | None = None


@dataclass(frozen=True)
class LineSearchResult:
    """The chosen point, if any, and the evaluated candidates in order."""

    selected: LineSearchTrial | None
    trials: tuple[LineSearchTrial, ...]
    failure_reason: str | None = None


@dataclass(frozen=True)
class ParabolaInterpolation:
    """Quadratic interpolant in the scaled coordinate t = step_size / beta."""

    beta: float
    f0: float
    f1: float
    f2: float

    @property
    def denominator(self) -> float:
        return (self.f0 - self.f1) + (self.f2 - self.f1)

    @property
    def numerator(self) -> float:
        return 3.0 * (self.f0 - self.f1) + (self.f2 - self.f1)

    def estimate_minimum(self) -> tuple[float | None, str | None]:
        """Return the fitted step and any reason why it cannot be used."""

        denominator = self.denominator
        if not isfinite(denominator):
            return None, "The denominator D is not finite."
        tolerance = 8 * np.finfo(float).eps * max(abs(self.f0), abs(self.f1), abs(self.f2))
        if abs(denominator) <= tolerance:
            return None, "The denominator D is numerically zero."
        if denominator < 0:
            return None, "D is negative; the interpolant has a maximum."
        fitted_beta = (self.beta / 2.0) * (self.numerator / denominator)
        if not isfinite(fitted_beta) or fitted_beta <= 0:
            return fitted_beta, "The fitted step β* must be finite and positive."
        return fitted_beta, None

    def evaluate(self, step_sizes: np.ndarray) -> np.ndarray:
        """Evaluate the interpolant without forming coefficients in 1 / beta²."""

        t = step_sizes / self.beta
        return self.f0 + t * ((self.f1 - self.f0) + 0.5 * self.denominator * (t - 1.0))


@dataclass(frozen=True, kw_only=True)
class ParabolaSearchResult(LineSearchResult):
    """The three samples, fitted candidate and decision for a single iteration."""

    start: LineSearchTrial
    interpolation: ParabolaInterpolation | None = None
    beta_star: float | None = None
    fitted: LineSearchTrial | None = None
    fit_issue: str | None = None

    @property
    def samples(self) -> tuple[LineSearchTrial, ...]:
        return (self.start, *self.trials[:2])


def successive_halving(
    objective: Callable[[np.ndarray], float],
    point: ArrayLike,
    gradient: ArrayLike,
    beta: float = 1.0,
    *,
    halving_factor: float = 0.5,
    max_halvings: int = DEFAULT_MAX_HALVINGS,
    base_value: float | None = None,
    max_trials: int | None = None,
) -> LineSearchResult:
    """Try x - beta * gradient with a fixed gradient until the objective decreases.

    The initial trial is followed by at most max_halvings reductions.
    The existing max_trials keyword counts the initial trial as well.
    """

    if not 0 < halving_factor < 1:
        raise ValueError("The halving factor must be between 0 and 1.")
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("At least one line-search trial is required.")
        max_halvings = max_trials - 1
    if max_halvings < 0:
        raise ValueError("Maximum halvings must be non-negative.")

    point, gradient, base_value = _line_search_inputs(
        objective, point, gradient, beta, base_value
    )
    current_point = tuple(point)
    trials: list[LineSearchTrial] = []
    for index in range(max_halvings + 1):
        phase = "initial beta" if index == 0 else "halving"
        trial = _trial(
            objective, point, gradient, beta, base_value, phase, require_movement=True
        )
        trials.append(trial)
        if trial.accepted:
            return LineSearchResult(trial, tuple(trials))
        if trial.point == current_point:
            return LineSearchResult(
                None, tuple(trials), "β is too small to change the current point."
            )
        smaller_beta = beta * halving_factor
        if index < max_halvings and not 0 < smaller_beta < beta:
            return LineSearchResult(
                None, tuple(trials), "β cannot be reduced further at floating-point precision."
            )
        beta = smaller_beta

    reason = f"The limit of {max_halvings} halvings was reached."
    if all(trial.value is None for trial in trials):
        reason += " Every test point or function value was invalid."
    return LineSearchResult(None, tuple(trials), reason)


def parabola_minimum(beta: float, f0: float, f1: float, f2: float) -> float | None:
    """Calculate beta* from objective values at 0, beta and 2*beta."""

    fitted_beta, issue = ParabolaInterpolation(beta, f0, f1, f2).estimate_minimum()
    return fitted_beta if issue is None else None


def parabola_fitting(
    objective: Callable[[np.ndarray], float],
    point: ArrayLike,
    gradient: ArrayLike,
    beta: float = DEFAULT_PARABOLA_BETA,
    *,
    base_value: float | None = None,
) -> ParabolaSearchResult:
    """Compare beta with the minimum of the fitted parabola."""

    point, gradient, base_value = _line_search_inputs(
        objective, point, gradient, beta, base_value
    )
    start = LineSearchTrial(0.0, tuple(map(float, point)), base_value, False, "start")
    first = _trial(
        objective, point, gradient, beta, base_value, "initial beta", require_movement=True
    )
    second = _trial(objective, point, gradient, 2.0 * beta, base_value, "second sample (2β)")
    trials = [first, second]
    interpolation: ParabolaInterpolation | None = None
    fitted_beta: float | None = None
    fitted: LineSearchTrial | None = None
    fit_issue: str | None = "An interpolation sample has an invalid function value."
    if first.value is not None and second.value is not None:
        interpolation = ParabolaInterpolation(beta, base_value, first.value, second.value)
        fitted_beta, fit_issue = interpolation.estimate_minimum()
        if fitted_beta is not None and fit_issue is None:
            if fitted_beta == beta:
                fitted = LineSearchTrial(
                    beta, first.point, first.value, first.accepted, "parabola fit",
                    first.rejection_reason,
                )
            else:
                fitted = _trial(
                    objective, point, gradient, fitted_beta, base_value, "parabola fit",
                    require_movement=True,
                )
            trials.append(fitted)
            fit_issue = fitted.rejection_reason

    # min preserves the first candidate on a tie. The 2β sample is for the fit only.
    improving = [
        trial for trial in (first, fitted)
        if trial is not None and trial.value is not None and trial.accepted
    ]
    selected = min(improving, key=lambda trial: float(trial.value), default=None)
    failure = None
    if selected is None:
        failure = "No improving step was found."
        if fit_issue:
            failure = f"{failure} {fit_issue}"
    return ParabolaSearchResult(
        selected=selected, trials=tuple(trials), failure_reason=failure, start=start,
        interpolation=interpolation, beta_star=fitted_beta, fitted=fitted, fit_issue=fit_issue,
    )


def _line_search_inputs(
    objective: Callable[[np.ndarray], float],
    point: ArrayLike,
    gradient: ArrayLike,
    beta: float,
    base_value: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isfinite(beta) or beta <= 0:
        raise ValueError("Beta must be finite and greater than zero.")
    point = np.asarray(point, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if point.ndim != 1 or gradient.shape != point.shape:
        raise ValueError("Point and gradient must be vectors of the same length.")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(gradient)):
        raise ValueError("Current point and gradient must be finite.")
    value = float(objective(point)) if base_value is None else base_value
    if not isfinite(value):
        raise ValueError("The objective must be finite at the starting point.")
    return point, gradient, value


def _trial(
    objective: Callable[[np.ndarray], float],
    point: np.ndarray,
    gradient: np.ndarray,
    beta: float,
    reference: float,
    phase: str,
    *,
    require_movement: bool = False,
) -> LineSearchTrial:
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        candidate = point - beta * gradient
        coordinates = tuple(map(float, candidate))
        if require_movement and np.array_equal(candidate, point):
            return LineSearchTrial(
                beta, coordinates, reference, False, phase, "Test point equals current point"
            )
        if not np.all(np.isfinite(candidate)):
            return LineSearchTrial(beta, coordinates, None, False, phase, "Invalid test point")
        try:
            value = float(objective(candidate))
        except (ValueError, TypeError, ArithmeticError):
            value = float("nan")
    if not isfinite(value):
        return LineSearchTrial(beta, coordinates, None, False, phase, "Invalid function value")
    return LineSearchTrial(beta, coordinates, value, value < reference, phase)
