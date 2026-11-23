"""Exact two-dimensional geometry shared by the hull and cutting-plane tools."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Iterator

from topics.simplex_method.solver import Inequality, Vector


Interval = tuple[Fraction | None, Fraction | None]
ZERO = Fraction(0)


@dataclass(frozen=True)
class HullStep:
    chain: str
    action: str
    point: Vector
    candidate: Vector | None = None
    turn: Fraction | None = None


@dataclass(frozen=True)
class HullConstruction:
    points: tuple[Vector, ...]
    vertices: tuple[Vector, ...]
    steps: tuple[HullStep, ...]

    def chains_at(self, index: int) -> tuple[tuple[Vector, ...], tuple[Vector, ...]]:
        chains: dict[str, list[Vector]] = {"Lower": [], "Upper": []}
        for step in self.steps[:index]:
            chain = chains[step.chain]
            if step.action == "add":
                chain.append(step.point)
            elif step.action == "remove":
                chain.pop()
        return tuple(chains["Lower"]), tuple(chains["Upper"])


def construct_hull(points: tuple[Vector, ...]) -> HullConstruction:
    steps: list[HullStep] = []
    vertices = convex_hull(points, trace=steps)
    return HullConstruction(points, vertices, tuple(steps))


def dot(left: Vector, right: Vector) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), ZERO)


def determinant(left: Vector, right: Vector) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _build_chain(points: Iterable[Vector], name: str, chain: list[Vector]) -> Iterator[HullStep]:
    for point in points:
        yield HullStep(name, "consider", point)
        while len(chain) >= 2:
            a, b = chain[-2:]
            turn = determinant((b[0] - a[0], b[1] - a[1]), (point[0] - a[0], point[1] - a[1]))
            if turn > 0:
                break
            yield HullStep(name, "remove", chain.pop(), point, turn)
        chain.append(point)
        yield HullStep(name, "add", point)


def convex_hull(points: Iterable[Vector], *, trace: list[HullStep] | None = None) -> tuple[Vector, ...]:
    """Monotone chain with exact turns. Returns CCW vertices without duplicates."""

    ordered = sorted(set(points))
    lower: list[Vector] = []
    upper: list[Vector] = []
    for entries, name, chain in ((ordered, "Lower", lower), (reversed(ordered), "Upper", upper)):
        events = _build_chain(entries, name, chain)
        if trace is not None:
            trace.extend(events)
        else:
            for _ in events:
                pass
    return tuple(lower[:-1] + upper[:-1]) if len(ordered) > 1 else tuple(ordered)


def feasible_vertices(rows: tuple[Inequality, ...]) -> tuple[Vector, ...]:
    """Intersect independent pairs of boundary lines and retain feasible points."""

    vertices = {}
    for left, right in combinations(rows, 2):
        a, b = left.coefficients
        c, d = right.coefficients
        det = determinant(left.coefficients, right.coefficients)
        if det:
            point = ((left.bound * d - b * right.bound) / det,
                     (a * right.bound - left.bound * c) / det)
            if point not in vertices and all(dot(row.coefficients, point) <= row.bound for row in rows):
                vertices[point] = None
    return tuple(vertices)


def _bounds(rows: Iterable[tuple[Fraction, Fraction]]) -> Interval | None:
    lower = upper = None
    for coefficient, bound in rows:
        if coefficient > 0:
            upper = bound / coefficient if upper is None else min(upper, bound / coefficient)
        elif coefficient < 0:
            lower = bound / coefficient if lower is None else max(lower, bound / coefficient)
        elif bound < 0:
            return None
    if lower is not None and upper is not None and lower > upper:
        return None
    return lower, upper


def axis_interval(rows: tuple[Inequality, ...], axis: int) -> Interval | None:
    """Project the region onto an axis by eliminating the other coordinate."""

    upper, lower, projected = [], [], []
    for row in rows:
        factor = row.coefficients[1 - axis]
        if factor == 0:
            projected.append((row.coefficients[axis], row.bound))
        else:
            normalized = (row.coefficients[axis] / abs(factor), row.bound / abs(factor))
            (upper if factor > 0 else lower).append(normalized)
    return _bounds([*projected, *((a + b, u + v) for a, u in upper for b, v in lower)])


def slice_interval(rows: tuple[Inequality, ...], axis: int, value: Fraction) -> Interval | None:
    """Find the interval of the other coordinate at a fixed axis value."""

    return _bounds(
        (row.coefficients[1 - axis], row.bound - value * row.coefficients[axis])
        for row in rows
    )


def choose_coordinate(interval: Interval) -> Fraction:
    low, high = interval
    if low is not None and low > 0:
        return low
    if high is not None and high < 0:
        return high
    return ZERO
