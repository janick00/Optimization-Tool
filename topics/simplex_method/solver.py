"""Exact start-basis validation and simplex pivots on E x <= b."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from sympy import Matrix

from topics.linear_programming.models import (
    ConstraintRelation, LinearProgram, ObjectiveSense, VariableDomain, _as_fraction,
)


Vector = tuple[Fraction, ...]
MAX_PIVOTS = 500


@dataclass(frozen=True)
class Inequality:
    name: str
    coefficients: Vector
    bound: Fraction
    is_domain: bool = False


@dataclass(frozen=True)
class StartCheck:
    row: Inequality
    value: Fraction

    @property
    def satisfied(self) -> bool:
        return self.value <= self.row.bound

    @property
    def active(self) -> bool:
        return self.value == self.row.bound


@dataclass(frozen=True)
class StartBasis:
    point: Vector
    checks: tuple[StartCheck, ...]
    basis: tuple[int, ...]
    active_rank: int
    objective_value: Fraction
    error: str | None = None

    @property
    def matrix(self) -> tuple[Vector, ...]:
        return tuple(self.checks[index].row.coefficients for index in self.basis)


@dataclass(frozen=True)
class SimplexStep:
    point: Vector
    basis: tuple[int, ...]
    released: int | None = None
    added: int | None = None
    distance: Fraction | None = None


@dataclass(frozen=True)
class PivotCalculation:
    step: SimplexStep
    inverse: tuple[Vector, ...]
    gains: tuple[tuple[int, Fraction], ...]
    direction: Vector | None
    ratios: tuple[tuple[Fraction, int], ...]
    decision: str


@dataclass(frozen=True)
class SimplexPhase:
    name: str
    rows: tuple[Inequality, ...]
    objective: Vector
    calculations: tuple[PivotCalculation, ...]


@dataclass(frozen=True)
class SimplexResult:
    mode: str
    start: StartBasis | None
    steps: tuple[SimplexStep, ...]
    status: str
    message: str
    phases: tuple[SimplexPhase, ...] = ()


def inequalities(problem: LinearProgram) -> tuple[Inequality, ...]:
    """Use input order followed by domain bounds, all written as <= rows."""

    rows: list[Inequality] = []
    for i, constraint in enumerate(problem.constraints, 1):
        equal = constraint.relation is ConstraintRelation.EQUAL
        if constraint.relation is not ConstraintRelation.GREATER_EQUAL:
            name = f"C{i} (≤)" if equal else f"C{i}"
            rows.append(Inequality(name, constraint.coefficients, constraint.right_hand_side))
        if constraint.relation is not ConstraintRelation.LESS_EQUAL:
            name = f"C{i} (≥)" if equal else f"C{i}"
            rows.append(Inequality(name, tuple(-c for c in constraint.coefficients),
                                   -constraint.right_hand_side))
    for i, (name, domain) in enumerate(zip(problem.variable_names, problem.variable_domains)):
        if domain is VariableDomain.FREE:
            continue
        nonnegative = domain is VariableDomain.NONNEGATIVE
        row = tuple(
            Fraction(-1 if nonnegative else 1) if index == i else Fraction(0)
            for index in range(len(problem.variable_names))
        )
        rows.append(Inequality(f"{name} {'≥' if nonnegative else '≤'} 0", row, Fraction(0), True))
    return tuple(rows)


def objective_value(problem: LinearProgram, point: Vector) -> Fraction:
    return _dot(problem.objective, point)


def solve_simplex(
    problem: LinearProgram,
    manual_start: Sequence[Fraction | int | float | str] | None = None,
    *,
    capture_details: bool = False,
) -> SimplexResult:
    """Start exactly at the supplied vertex, or find one via a feasibility phase."""

    rows = inequalities(problem)
    dimension = len(problem.objective)
    phases: list[SimplexPhase] | None = [] if capture_details else None
    if manual_start is None:
        mode = "Automatic"
        point, error, failure_status = _automatic_start(rows, dimension, phases)
        if point is None:
            return SimplexResult(
                mode, None, (), failure_status, error or "No starting vertex found.", tuple(phases or ()),
            )
    else:
        mode = "Manual"
        point = tuple(_as_fraction(value) for value in manual_start)
        if len(point) != dimension:
            raise ValueError(f"Starting Point requires exactly {dimension} coordinates.")

    start = _check_start(problem, rows, point)
    if start.error is not None:
        return SimplexResult(mode, start, (), "error", start.error)

    objective = problem.objective
    if problem.objective_sense is ObjectiveSense.MINIMIZE:
        objective = tuple(-c for c in objective)
    steps, status = _run_phase("Phase II · objective", rows, objective, SimplexStep(point, start.basis), phases)
    messages = {
        "optimal": "Optimal vertex found.",
        "unbounded": "The objective is unbounded along an improving direction.",
        "limit": f"Stopped after the limit of {MAX_PIVOTS} simplex pivots.",
    }
    return SimplexResult(mode, start, steps, status, messages[status], tuple(phases or ()))


def _check_start(problem: LinearProgram, rows: tuple[Inequality, ...], point: Vector) -> StartBasis:
    checks = tuple(StartCheck(row, _dot(row.coefficients, point)) for row in rows)
    active = tuple(i for i, check in enumerate(checks) if check.active)
    independent = _independent_rows(rows, active, len(point))
    rank = len(independent)
    violated_constraints = [check.row.name for check in checks if not check.satisfied and not check.row.is_domain]
    violated_domains = [check.row.name for check in checks if not check.satisfied and check.row.is_domain]
    issues = []
    if violated_constraints:
        issues.append("Violated constraints: " + ", ".join(violated_constraints) + ".")
    if violated_domains:
        issues.append("Violated variable domains: " + ", ".join(violated_domains) + ".")
    if not issues and rank < len(point):
        issues.append(
            f"The point is feasible but is not a vertex. Active rank {rank}; required rank {len(point)}."
        )
    return StartBasis(
        point, checks, independent if not issues else (), rank, objective_value(problem, point),
        " ".join(issues) if issues else None,
    )


def _automatic_start(
    rows: tuple[Inequality, ...], dimension: int, phases: list[SimplexPhase] | None
) -> tuple[Vector | None, str | None, str]:
    if len(_independent_rows(rows, tuple(range(len(rows))), dimension)) < dimension:
        return None, "The constraint and domain normals do not span the space. No full-rank vertex basis exists.", "error"

    origin = (Fraction(0),) * dimension
    artificial_value = max((Fraction(0), *(-row.bound for row in rows)))
    if artificial_value == 0:
        return _vertex_from_feasible(rows, origin).point, None, ""

    # Phase I: minimize the shared violation t in E x - t <= b, t >= 0.
    augmented = tuple(
        Inequality(row.name, (*row.coefficients, Fraction(-1)), row.bound) for row in rows
    ) + (Inequality("t ≥ 0", (*origin, Fraction(-1)), Fraction(0)),)
    phase_start = _vertex_from_feasible(augmented, (*origin, artificial_value))
    phase_steps, status = _run_phase(
        "Phase I · feasibility", augmented, (*origin, Fraction(-1)), phase_start, phases,
    )
    point = phase_steps[-1].point
    if point[-1] == 0:
        return point[:-1], None, ""
    if status == "optimal":
        return None, "The feasibility phase found no feasible point for these constraints and domains.", "infeasible"
    return None, "The feasibility phase stopped before it established a feasible starting vertex.", "limit"


def _vertex_from_feasible(rows: tuple[Inequality, ...], point: Vector) -> SimplexStep:
    """Move along an active nullspace until a new independent row blocks each move."""

    while True:
        active = tuple(i for i, row in enumerate(rows) if _dot(row.coefficients, point) == row.bound)
        basis = _independent_rows(rows, active, len(point))
        if len(basis) == len(point):
            return SimplexStep(point, basis)
        direction = _fractions(_matrix(rows, basis, len(point)).nullspace()[0])
        blocked = _blocking_row(rows, point, direction, ())
        if blocked is None:
            direction = tuple(-value for value in direction)
            blocked = _blocking_row(rows, point, direction, ())
        if blocked is None:
            raise ArithmeticError("Could not reach a vertex from the feasible point.")
        distance, _ = blocked
        point = tuple(x + distance * d for x, d in zip(point, direction))


def _run_phase(
    name: str, rows: tuple[Inequality, ...], objective: Vector, first: SimplexStep,
    phases: list[SimplexPhase] | None,
) -> tuple[tuple[SimplexStep, ...], str]:
    calculations: list[PivotCalculation] | None = [] if phases is not None else None
    result = _pivot(rows, objective, first, calculations)
    if phases is not None and calculations is not None:
        phases.append(SimplexPhase(name, rows, objective, tuple(calculations)))
    return result


def _pivot(
    rows: tuple[Inequality, ...], objective: Vector, first: SimplexStep,
    calculations: list[PivotCalculation] | None,
) -> tuple[tuple[SimplexStep, ...], str]:
    """Pivot on active bounds with Bland's index rule for both choices."""

    steps = [first]
    while True:
        current = steps[-1]
        inverse = _matrix(rows, current.basis, len(objective)).inv()
        gains = []
        for position, index in enumerate(current.basis):
            direction = tuple(-value for value in _fractions(inverse[:, position]))
            gain = _dot(objective, direction)
            if calculations is not None:
                gains.append((index, gain))
            if gain > 0:
                break
        else:
            if calculations is not None:
                calculations.append(PivotCalculation(
                    current, _inverse_rows(inverse), tuple(gains), None, (), "Optimal basis",
                ))
            return tuple(steps), "optimal"
        ratios = _ratios(rows, current.point, direction, current.basis)
        blocked = min(ratios, default=None)
        if calculations is not None:
            decision = (
                "Unbounded direction" if blocked is None else
                "Pivot limit" if len(steps) - 1 == MAX_PIVOTS else
                f"Release {rows[index].name}, add {rows[blocked[1]].name}, step {blocked[0]}"
            )
            calculations.append(PivotCalculation(
                current, _inverse_rows(inverse), tuple(gains), direction, tuple(ratios), decision,
            ))
        if blocked is None:
            return tuple(steps), "unbounded"
        if len(steps) - 1 == MAX_PIVOTS:
            return tuple(steps), "limit"
        distance, incoming = blocked
        point = tuple(x + distance * d for x, d in zip(current.point, direction))
        basis = tuple(sorted(incoming if old == index else old for old in current.basis))
        steps.append(SimplexStep(point, basis, index, incoming, distance))


def _blocking_row(
    rows: tuple[Inequality, ...], point: Vector, direction: Vector, excluded: tuple[int, ...]
) -> tuple[Fraction, int] | None:
    return min(_ratios(rows, point, direction, excluded), default=None)


def _ratios(
    rows: tuple[Inequality, ...], point: Vector, direction: Vector, excluded: tuple[int, ...]
) -> list[tuple[Fraction, int]]:
    ratios = []
    for index, row in enumerate(rows):
        if index in excluded:
            continue
        rate = _dot(row.coefficients, direction)
        if rate > 0:
            ratios.append(((row.bound - _dot(row.coefficients, point)) / rate, index))
    return ratios


def _independent_rows(
    rows: tuple[Inequality, ...], indices: tuple[int, ...], dimension: int
) -> tuple[int, ...]:
    _, pivot_columns = _matrix(rows, indices, dimension).T.rref()
    return tuple(indices[index] for index in pivot_columns)


def _matrix(rows: tuple[Inequality, ...], indices: tuple[int, ...], dimension: int) -> Matrix:
    return Matrix([rows[index].coefficients for index in indices]) if indices else Matrix(0, dimension, [])


def _fractions(values: Matrix) -> Vector:
    return tuple(Fraction(int(value.p), int(value.q)) for value in values)


def _inverse_rows(values: Matrix) -> tuple[Vector, ...]:
    return tuple(_fractions(values[row, :]) for row in range(values.rows))


def _dot(left: Vector, right: Vector) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))
