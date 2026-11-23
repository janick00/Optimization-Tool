"""Finite 2D integer and mixed-integer hulls, using exact rational arithmetic."""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor

from topics.integer_programming.hull_geometry import (
    Interval, axis_interval, choose_coordinate, convex_hull, feasible_vertices, slice_interval,
)
from topics.integer_programming.models import Domain
from topics.integer_programming.solver import relaxation
from topics.linear_programming.models import LinearProgram, ObjectiveSense
from topics.simplex_method.solver import Inequality, Vector, inequalities, objective_value, solve_simplex


@dataclass(frozen=True)
class Region:
    rows: tuple[Inequality, ...]
    vertices: tuple[Vector, ...]
    limits: tuple[Interval, Interval] | None

    @property
    def bounded(self) -> bool:
        return self.limits is not None and all(b is not None for pair in self.limits for b in pair)


@dataclass(frozen=True)
class Optimum:
    point: Vector
    value: Fraction


@dataclass(frozen=True)
class RelaxationResult:
    status: str
    optimum: Optimum | None = None


@dataclass(frozen=True)
class IntegerHull:
    status: str
    region: Region
    lp: RelaxationResult | None = None
    points: tuple[Vector, ...] = ()
    segments: tuple[tuple[Vector, Vector], ...] = ()
    vertices: tuple[Vector, ...] = ()
    discrete_optimum: Optimum | None = None
    hull_optimum: Optimum | None = None

    @property
    def gap(self) -> Fraction | None:
        lp = self.lp.optimum if self.lp else None
        if lp is not None and self.discrete_optimum is not None:
            return abs(lp.value - self.discrete_optimum.value)
        return None


def analyze_region(problem: LinearProgram) -> Region:
    """Classify the full region independently of the objective or plot axes."""

    if len(problem.objective) != 2:
        raise ValueError("Convex Hull / Cutting Planes requires exactly two variables.")
    rows = inequalities(problem)
    x, y = axis_interval(rows, 0), axis_interval(rows, 1)
    if x is None or y is None:
        return Region(rows, (), None)
    return Region(rows, convex_hull(feasible_vertices(rows)), (x, y))


def solve_relaxation(problem: LinearProgram, region: Region) -> RelaxationResult:
    """Use Simplex when a starting vertex exists, with exact 2D projection as fallback."""

    if region.limits is None:
        return RelaxationResult("infeasible")
    if region.vertices:
        solved = solve_simplex(problem, manual_start=region.vertices[0])
        if solved.status == "optimal":
            point = solved.steps[-1].point
            return RelaxationResult("optimal", Optimum(point, objective_value(problem, point)))
        if solved.status == "unbounded":
            return RelaxationResult("unbounded")

    # Change coordinates to (z = cᵀx, remaining coordinate) and project onto z.
    if not any(problem.objective):
        x = choose_coordinate(region.limits[0])
        interval = slice_interval(region.rows, 0, x)
        assert interval is not None
        return RelaxationResult("optimal", Optimum((x, choose_coordinate(interval)), Fraction(0)))
    fixed = 0 if problem.objective[0] else 1
    c, d = problem.objective[fixed], problem.objective[1 - fixed]
    transformed = tuple(
        Inequality(row.name, (row.coefficients[fixed] / c,
                              row.coefficients[1 - fixed] - row.coefficients[fixed] * d / c), row.bound)
        for row in region.rows
    )
    limits = axis_interval(transformed, 0)
    assert limits is not None
    value = limits[1] if problem.objective_sense is ObjectiveSense.MAXIMIZE else limits[0]
    if value is None:
        return RelaxationResult("unbounded")
    interval = slice_interval(transformed, 0, value)
    assert interval is not None
    other = choose_coordinate(interval)
    coordinate = (value - d * other) / c
    point = (coordinate, other) if fixed == 0 else (other, coordinate)
    return RelaxationResult("optimal", Optimum(point, value))


def _best(problem: LinearProgram, points: tuple[Vector, ...]) -> Optimum:
    select = max if problem.objective_sense is ObjectiveSense.MAXIMIZE else min
    point = select(sorted(points), key=lambda p: objective_value(problem, p))
    return Optimum(point, objective_value(problem, point))


def compute_integer_hull(
    problem: LinearProgram, domains: tuple[Domain, ...], *,
    search_radius: int = 100, max_points: int = 1500, solve_lp: bool = True,
) -> IntegerHull:
    """Enumerate bounded integer slices. Return no hull if coverage cannot be guaranteed."""

    if search_radius < 1 or max_points < 1:
        raise ValueError("Search radius and maximum points must be positive.")
    relaxed = relaxation(problem, domains)
    region = analyze_region(relaxed)
    lp = solve_relaxation(relaxed, region) if solve_lp else None
    if region.limits is None:
        return IntegerHull("empty_lp", region, lp)
    if not region.bounded:
        return IntegerHull("unbounded_region", region, lp)

    discrete = tuple(i for i, domain in enumerate(domains) if domain is not Domain.CONTINUOUS)
    if not discrete:
        optimum = _best(relaxed, region.vertices)
        return IntegerHull("complete", region, lp, vertices=region.vertices,
                           discrete_optimum=optimum, hull_optimum=optimum)
    integer_limits = {}
    for axis in discrete:
        low, high = region.limits[axis]
        assert low is not None and high is not None
        integer_limits[axis] = ceil(low), floor(high)
    if any(low > high for low, high in integer_limits.values()):
        return IntegerHull("empty_discrete", region, lp)
    if any(low < -search_radius or high > search_radius for low, high in integer_limits.values()):
        return IntegerHull("search_truncated", region, lp)

    fixed = min(discrete, key=lambda i: integer_limits[i][1] - integer_limits[i][0])
    low, high = integer_limits[fixed]
    points: list[Vector] = []
    segments: list[tuple[Vector, Vector]] = []
    count = 0
    for value in range(low, high + 1):
        fixed_value = Fraction(value)
        interval = slice_interval(region.rows, fixed, fixed_value)
        if interval is None:
            continue
        bottom, top = interval
        assert bottom is not None and top is not None
        if len(discrete) == 2:
            first, last = ceil(bottom), floor(top)
            count += max(0, last - first + 1)
            if count > max_points:
                return IntegerHull("point_limit", region, lp)
            for other in range(first, last + 1):
                pair = (fixed_value, Fraction(other))
                points.append(pair if fixed == 0 else pair[::-1])
        else:
            start, end = (fixed_value, bottom), (fixed_value, top)
            if fixed == 1:
                start, end = start[::-1], end[::-1]
            count += 1 if start == end else 2
            if count > max_points:
                return IntegerHull("point_limit", region, lp)
            if start == end:
                points.append(start)
            else:
                segments.append((start, end))
    generators = (*points, *(p for segment in segments for p in segment))
    if not generators:
        return IntegerHull("empty_discrete", region, lp)
    hull = convex_hull(generators)
    return IntegerHull("complete", region, lp, tuple(sorted(points)), tuple(segments), hull,
                       _best(relaxed, generators), _best(relaxed, hull))
