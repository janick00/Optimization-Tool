"""Settings and iteration history for continuous optimization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from topics.unconstrained_optimization.methods import LineSearchTrial, ParabolaSearchResult
from topics.unconstrained_optimization.methods.line_search import (
    DEFAULT_MAX_HALVINGS,
    DEFAULT_PARABOLA_BETA,
)
from topics.unconstrained_optimization.methods.newton_methods import BroydenUpdate
from topics.unconstrained_optimization.objective import ParsedFunction


class OptimizationMethod(str, Enum):
    """Available methods from the continuous-optimization lectures."""

    GRADIENT_HALVING = "gradient_halving"
    GRADIENT_PARABOLA = "gradient_parabola"
    NEWTON = "newton"
    BROYDEN = "broyden"

    @property
    def label(self) -> str:
        return {
            OptimizationMethod.GRADIENT_HALVING: "Gradient descent - successive halving",
            OptimizationMethod.GRADIENT_PARABOLA: "Gradient descent - direct parabola fitting",
            OptimizationMethod.NEWTON: "Newton's method",
            OptimizationMethod.BROYDEN: "Broyden's method",
        }[self]


@dataclass(frozen=True)
class OptimizationSettings:
    """Numerical settings shared by all algorithms."""

    initial_step: float = 1.0
    tolerance: float = 1e-6
    max_iterations: int = 30
    halving_factor: float = 0.5
    max_halvings: int = DEFAULT_MAX_HALVINGS

    @classmethod
    def for_method(cls, method: OptimizationMethod) -> OptimizationSettings:
        if method is OptimizationMethod.GRADIENT_HALVING:
            return cls(max_iterations=10)
        if method is OptimizationMethod.GRADIENT_PARABOLA:
            return cls(initial_step=DEFAULT_PARABOLA_BETA, max_iterations=100)
        return cls()

    def __post_init__(self) -> None:
        if not isfinite(self.initial_step) or self.initial_step <= 0:
            raise ValueError("The initial step size must be finite and greater than zero.")
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Tolerance must be finite and greater than zero.")
        if self.max_iterations < 1:
            raise ValueError("Maximum iterations must be at least one.")
        if not 0 < self.halving_factor < 1:
            raise ValueError("The halving factor must be between 0 and 1.")
        if self.max_halvings < 0:
            raise ValueError("Maximum halvings must be non-negative.")


@dataclass(frozen=True)
class SecondOrderDetails:
    """Matrix used for a Newton or Broyden step and Broyden's update vectors."""

    matrix: tuple[tuple[float, ...], ...] | None
    displacement: tuple[float, ...] | None = None
    gradient_change: tuple[float, ...] | None = None
    direction: tuple[float, ...] | None = None
    beta: float = 1.0
    candidate: tuple[float, ...] | None = None
    candidate_value: float | None = None
    update: BroydenUpdate | None = None
    update_note: str = ""


@dataclass(frozen=True)
class IterationStep:
    """All values recorded for one main optimization iteration."""

    iteration: int
    point: tuple[float, ...]
    value: float
    gradient: tuple[float, ...] | None
    gradient_norm: float | None
    step_size: float
    note: str
    trials: tuple[LineSearchTrial, ...] = ()
    parabola: ParabolaSearchResult | None = None
    second_order: SecondOrderDetails | None = None


@dataclass(frozen=True)
class OptimizationResult:
    """Optimization result with the complete intermediate history."""

    function: ParsedFunction
    method: OptimizationMethod
    steps: tuple[IterationStep, ...]
    converged: bool
    message: str
    failed_trials: tuple[LineSearchTrial, ...] = ()
    failed_parabola: ParabolaSearchResult | None = None
    first_convergence_iteration: int | None = None
    completed_requested_iterations: bool = False
    settings: OptimizationSettings = OptimizationSettings()
    failed_second_order: SecondOrderDetails | None = None
    stationary_hessian: tuple[tuple[float, ...], ...] | None = None
    initial_hessian: tuple[tuple[float, ...], ...] | None = None

    @property
    def convergence_iteration(self) -> int | None:
        if self.first_convergence_iteration is not None:
            return self.first_convergence_iteration
        return self.steps[-1].iteration if self.converged else None

    @property
    def parabola_iterations(self) -> tuple[tuple[IterationStep, ParabolaSearchResult], ...]:
        iterations = [
            (current, following.parabola)
            for current, following in zip(self.steps, self.steps[1:])
            if following.parabola is not None
        ]
        if self.failed_parabola is not None:
            iterations.append((self.steps[-1], self.failed_parabola))
        return tuple(iterations)

    @property
    def accepted_halving_step(self) -> IterationStep | None:
        if self.method is OptimizationMethod.GRADIENT_HALVING and len(self.steps) > 1:
            return self.steps[-1]
        return None

    @property
    def final_point(self) -> tuple[float, ...]:
        return self.steps[-1].point

    @property
    def final_value(self) -> float:
        return self.steps[-1].value
