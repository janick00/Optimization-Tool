"""Exact evaluation of user-chosen mixed-integer feasible points."""

from dataclasses import dataclass
from fractions import Fraction

from topics.integer_programming.models import Bounds, Domain
from topics.linear_programming.models import ConstraintRelation, LinearProgram, ObjectiveSense, VariableDomain
from topics.simplex_method.solver import SimplexResult, Vector, objective_value


@dataclass(frozen=True)
class PointCheck:
    point: Vector
    value: Fraction | None
    integrality_errors: tuple[str, ...] = ()
    domain_errors: tuple[str, ...] = ()
    constraint_errors: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.value is not None


def check_integer_point(problem: LinearProgram, domains: tuple[Domain, ...], point: Vector) -> PointCheck:
    """Check the original LP rows, sign bounds and declared integrality."""

    if len(point) != len(problem.objective) or len(domains) != len(point):
        raise ValueError(f"The point requires exactly {len(problem.objective)} coordinates.")
    integrality, bounds, constraints = [], [], []
    for name, domain, sign, value in zip(problem.variable_names, domains, problem.variable_domains, point):
        if domain is Domain.BINARY and value not in (0, 1):
            target = integrality if 0 <= value <= 1 else bounds
            target.append(f"{name} = {value} must be 0 or 1 (Binary).")
        elif domain is Domain.INTEGER and value.denominator != 1:
            integrality.append(f"{name} = {value} must be an integer.")
        if sign is VariableDomain.NONNEGATIVE and value < 0:
            bounds.append(f"{name} = {value} violates {name} ≥ 0.")
        elif sign is VariableDomain.NONPOSITIVE and value > 0:
            bounds.append(f"{name} = {value} violates {name} ≤ 0.")

    for i, row in enumerate(problem.constraints, 1):
        left = sum(a * x for a, x in zip(row.coefficients, point))
        right = row.right_hand_side
        if row.relation is ConstraintRelation.LESS_EQUAL:
            satisfied, symbol = left <= right, "≤"
        elif row.relation is ConstraintRelation.GREATER_EQUAL:
            satisfied, symbol = left >= right, "≥"
        else:
            satisfied, symbol = left == right, "="
        if not satisfied:
            constraints.append(f"C{i}: {left} {symbol} {right} is violated.")

    value = None if integrality or bounds or constraints else objective_value(problem, point)
    return PointCheck(point, value, tuple(integrality), tuple(bounds), tuple(constraints))


def relaxation_value(problem: LinearProgram, result: SimplexResult) -> Fraction | None:
    return objective_value(problem, result.steps[-1].point) if result.status == "optimal" else None


def relaxation_bounds(problem: LinearProgram, result: SimplexResult, comparison: PointCheck | None) -> Bounds:
    """Combine the certified LP bound with a feasible comparison value."""

    lp = relaxation_value(problem, result)
    manual = comparison.value if comparison is not None else None
    if problem.objective_sense is ObjectiveSense.MAXIMIZE:
        return Bounds(lower=manual, upper=lp)
    return Bounds(lower=lp, upper=manual)
