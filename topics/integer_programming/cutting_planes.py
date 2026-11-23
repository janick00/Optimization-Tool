"""Certified two-dimensional Gomory–Chvátal cuts and replayable solver steps."""

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from math import floor, gcd, lcm

from topics.integer_programming.hull_geometry import determinant, dot
from topics.integer_programming.integer_hull import (
    IntegerHull, Region, RelaxationResult, analyze_region, compute_integer_hull, solve_relaxation,
)
from topics.integer_programming.models import Domain
from topics.integer_programming.solver import branch_and_bound, relaxation
from topics.linear_programming.models import ConstraintRelation, LinearConstraint, LinearProgram, ObjectiveSense
from topics.simplex_method.solver import Inequality, Vector


@dataclass(frozen=True)
class Cut:
    coefficients: Vector
    bound: Fraction
    combination: tuple[tuple[Fraction, Inequality], ...] = ()
    combined_bound: Fraction | None = None
    validation: str = ""


def _weighted_rows(
    target: Vector, active: tuple[Inequality, ...],
) -> tuple[tuple[Fraction, Inequality], ...]:
    # A direction in a two-dimensional cone needs at most two generating rows.
    for index, left in enumerate(active):
        a = left.coefficients
        coordinate = 0 if a[0] else 1
        factor = target[coordinate] / a[coordinate]
        if factor > 0 and tuple(factor * v for v in a) == target:
            return ((factor, left),)
        for right in active[index + 1:]:
            b = right.coefficients
            det = determinant(a, b)
            if det:
                u = determinant(target, b) / det
                v = determinant(a, target) / det
                if u > 0 and v > 0:
                    return (u, left), (v, right)
    return ()


def gomory_chvatal_cut(
    region: Region, domains: tuple[Domain, ...], point: Vector,
) -> Cut | None:
    """Look for an integral bound supported by rows tight at the LP optimum.

    Continuous coordinates must cancel in the combination. The resulting
    left-hand side is then integer at every feasible mixed-integer point.
    """

    active = tuple(row for row in region.rows
                   if any(row.coefficients) and dot(row.coefficients, point) == row.bound)
    # Axes, short diagonals, then primitive integer directions of the active rows.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    primitive = []
    for row in active:
        scale = lcm(*(c.denominator for c in row.coefficients))
        a, b = (int(c * scale) for c in row.coefficients)
        divisor = gcd(a, b)
        primitive.append((a // divisor, b // divisor))
    directions.extend(primitive)
    for a, b in dict.fromkeys(directions):
        target = (Fraction(a), Fraction(b))
        if any(c and domain is Domain.CONTINUOUS for c, domain in zip(target, domains)):
            continue
        value = dot(target, point)
        if value.denominator == 1:
            continue
        weighted = _weighted_rows(target, active)
        if weighted:
            return _round_combination(target, weighted)

    # If both axes are discrete, search inside any full-dimensional active cone.
    # A large multiple of an integral interior direction plus a fractional
    # coordinate's unit vector stays in the cone and has fractional value.
    if Domain.CONTINUOUS not in domains:
        axis = next((i for i, value in enumerate(point) if value.denominator != 1), None)
        if axis is not None:
            unit = (Fraction(axis == 0), Fraction(axis == 1))
            for a, b in combinations(primitive, 2):
                det = determinant(a, b)
                if det:
                    interior = tuple(Fraction(u + v) for u, v in zip(a, b))
                    factor = 1 + floor(max(Fraction(0), -determinant(unit, b) / det,
                                           -determinant(a, unit) / det))
                    target = (interior if dot(interior, point).denominator != 1 else
                              tuple(factor * v + e for v, e in zip(interior, unit)))
                    weighted = _weighted_rows(target, active)
                    if weighted:
                        return _round_combination(target, weighted)
    return None


def _round_combination(target: Vector, weighted: tuple[tuple[Fraction, Inequality], ...]) -> Cut:
    bound = sum((weight * row.bound for weight, row in weighted), Fraction(0))
    return Cut(target, Fraction(floor(bound)), weighted, bound)


def validate_manual_cut(
    problem: LinearProgram, domains: tuple[Domain, ...], reference: IntegerHull, point: Vector,
    coefficients: Vector, bound: Fraction,
) -> Cut:
    """Accept a separating inequality only when its validity is proved on the original ILP."""

    if len(coefficients) != 2 or not any(coefficients):
        raise ValueError("Enter two coefficients, with at least one nonzero coefficient.")
    witness = None
    if reference.status == "complete":
        witness = max(reference.vertices, key=lambda p: dot(coefficients, p))
        method = "Maximum checked over the complete integer hull"
    elif reference.status in ("empty_lp", "empty_discrete"):
        method = "The original integer-feasible set is empty"
    else:
        check_problem = replace(problem, objective=coefficients, objective_sense=ObjectiveSense.MAXIMIZE)
        check = branch_and_bound(check_problem, domains)
        if check.status == "optimal":
            incumbent = check.events[-1].incumbent
            assert incumbent is not None
            witness = incumbent.point
            method = "Maximum over the original integer-feasible set proved by Branch and Bound"
        elif check.status == "infeasible":
            method = "Branch and Bound proved that the original integer-feasible set is empty"
        else:
            raise ValueError("Validity could not be proved for the complete integer-feasible set.")

    if witness is not None:
        value = dot(coefficients, witness)
        if value > bound:
            coordinates = ", ".join(map(str, witness))
            raise ValueError(f"Invalid cut. The feasible point ({coordinates}) gives {value} > {bound}.")
        method = f"{method}, with maximum {value} ≤ {bound}"
    if dot(coefficients, point) <= bound:
        raise ValueError("The inequality does not cut off the current LP optimum.")
    return Cut(coefficients, bound, validation=method + ".")


@dataclass(frozen=True)
class Comparison:
    cut: Cut | None
    lp: RelaxationResult


@dataclass(frozen=True)
class CutFrame:
    phase: str
    problem: LinearProgram
    region: Region
    lp: RelaxationResult | None = None
    comparisons: tuple[Comparison, ...] = ()
    cuts: tuple[Cut, ...] = ()
    fractional: tuple[int, ...] = ()
    candidate: Cut | None = None
    message: str = ""


class CuttingSession:
    """Advance one action at a time. Navigation replays previously computed frames."""

    def __init__(self, problem: LinearProgram, domains: tuple[Domain, ...], **limits: int) -> None:
        self.domains = domains
        self.source = problem
        self.reference = compute_integer_hull(problem, domains, solve_lp=False, **limits)
        relaxed = relaxation(problem, domains)
        self._frames = [CutFrame("region", relaxed, self.reference.region)]
        self.index = 0

    @property
    def frame(self) -> CutFrame:
        return self._frames[self.index]

    @property
    def can_advance(self) -> bool:
        return self.index < len(self._frames) - 1 or self.frame.phase not in ("finished", "no_cut")

    @property
    def can_add_manual(self) -> bool:
        return self.frame.phase in ("checked", "no_cut", "preview") and bool(self.frame.fractional)

    def reset(self) -> None:
        del self._frames[1:]
        self.index = 0

    def previous(self) -> None:
        self.index = max(0, self.index - 1)

    def next(self) -> None:
        if self.index < len(self._frames) - 1:
            self.index += 1
        elif self.can_advance:
            self._append(self._advance())

    def add_manual_cut(self, coefficients: Vector, bound: Fraction) -> None:
        if not self.can_add_manual:
            raise ValueError("Check integrality at a fractional LP optimum before entering a cut.")
        frame = self.frame
        assert frame.lp and frame.lp.optimum
        cut = validate_manual_cut(self.source, self.domains, self.reference, frame.lp.optimum.point,
                                  coefficients, bound)
        preview = replace(frame, phase="preview", candidate=cut, message="")
        del self._frames[self.index + 1:]
        if frame.phase == "preview":
            self._frames[self.index] = preview
        else:
            self._append(preview)

    def _append(self, frame: CutFrame) -> None:
        self._frames.append(frame)
        self.index += 1

    def _advance(self) -> CutFrame:
        frame = self.frame
        if frame.phase in ("region", "applied"):
            lp = solve_relaxation(frame.problem, frame.region)
            entry = Comparison(frame.cuts[-1] if frame.cuts else None, lp)
            return replace(frame, phase="solved", lp=lp, comparisons=(*frame.comparisons, entry))
        if frame.phase == "preview":
            cut = frame.candidate
            assert cut is not None
            row = LinearConstraint(cut.coefficients, ConstraintRelation.LESS_EQUAL, cut.bound,
                                   f"Cut {len(frame.cuts) + 1}")
            problem = replace(frame.problem, constraints=(*frame.problem.constraints, row))
            return CutFrame("applied", problem, analyze_region(problem),
                            comparisons=frame.comparisons, cuts=(*frame.cuts, cut))
        assert frame.lp is not None
        if frame.lp.optimum is None:
            message = ("The current relaxation is empty, proving integer infeasibility."
                       if frame.lp.status == "infeasible" else
                       "The LP objective is unbounded. This step sequence needs a finite LP optimum.")
            return replace(frame, phase="finished", message=message)
        point = frame.lp.optimum.point
        if frame.phase == "solved":
            fractional = tuple(i for i, domain in enumerate(self.domains)
                               if domain is not Domain.CONTINUOUS and point[i].denominator != 1)
            return replace(frame, phase="checked", fractional=fractional)
        if not frame.fractional:
            return replace(frame, phase="finished",
                           message="The LP optimum satisfies all variable domains and proves optimality.")
        cut = gomory_chvatal_cut(frame.region, self.domains, point)
        if cut is None:
            return replace(frame, phase="no_cut",
                           message="No supported Gomory–Chvátal direction separates this LP optimum. "
                                   "You can try a manual inequality.")
        return replace(frame, phase="preview", candidate=cut)
