"""Connector inequalities, exact small LPs and binary spanning solutions."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import Iterator

from sympy import Matrix

from topics.integer_programming.hull_geometry import dot
from topics.linear_programming.models import (
    ConstraintRelation, LinearConstraint, LinearProgram, ObjectiveSense, VariableDomain,
)
from topics.network_optimization.models import Edge, Graph, Method, SpanningState
from topics.network_optimization.spanning import _Components, minimum_spanning
from topics.simplex_method.solver import Vector, inequalities, objective_value, solve_simplex


FULL_CUT_NODES = 9
FULL_PARTITION_NODES = 5
EXACT_LP_EDGES = 16
ENUMERATION_EDGES = 12
ZERO = Fraction(0)
Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ConnectorRow:
    groups: Partition
    indices: tuple[int, ...]

    @property
    def rhs(self) -> int:
        return len(self.groups) - 1

    def activity(self, point: Vector) -> Fraction:
        return sum((point[i] for i in self.indices), ZERO)


@dataclass(frozen=True)
class ConnectorModels:
    graph: Graph
    edges: tuple[Edge, ...]
    cuts: tuple[ConnectorRow, ...]
    additions: tuple[ConnectorRow, ...]
    cuts_complete: bool
    partitions_complete: bool
    connected: bool

    @property
    def partition_rows(self) -> tuple[ConnectorRow, ...]:
        return self.cuts + self.additions


@dataclass(frozen=True)
class ConnectorLP:
    status: str
    point: Vector | None = None
    bound: Fraction | None = None


@dataclass(frozen=True)
class ConnectorOptima:
    connector: frozenset[Edge]
    connector_cost: Fraction
    tree: frozenset[Edge]
    tree_cost: Fraction
    verification: str


def set_partitions(size: int) -> Iterator[Partition]:
    """Yield unlabeled partitions exactly once, inserting vertices in order."""

    groups: list[list[int]] = []

    def extend(node: int) -> Iterator[Partition]:
        if node == size:
            yield tuple(tuple(group) for group in groups)
            return
        for group in groups:
            group.append(node)
            yield from extend(node + 1)
            group.pop()
        groups.append([node])
        yield from extend(node + 1)
        groups.pop()

    yield from extend(0)


def connector_row(edges: tuple[Edge, ...], groups: Partition) -> ConnectorRow:
    memberships = {node: group for group, vertices in enumerate(groups) for node in vertices}
    return ConnectorRow(groups, tuple(i for i, (u, v) in enumerate(edges) if memberships[u] != memberships[v]))


def _connected(size: int, edges) -> bool:
    components = _Components(size)
    for edge in edges:
        components.join(*edge)
    return components.count == 1


def build_connector_models(graph: Graph, *, illustration: Partition | None = None) -> ConnectorModels:
    """Generate full systems for small graphs and valid subsets for larger graphs."""

    if graph.directed:
        raise ValueError("The connector formulation requires an undirected graph.")
    size = graph.size
    edges = graph.edges()
    vertices = tuple(range(size))
    if illustration is not None and (
        len(illustration) < 3 or any(not group for group in illustration)
        or sorted(node for group in illustration for node in group) != list(vertices)
    ):
        raise ValueError("The illustrated partition needs at least three groups covering the vertices once.")
    full_cuts = size <= FULL_CUT_NODES
    full_partitions = size <= FULL_PARTITION_NODES
    if full_cuts:
        # Fix vertex 0 on one side, since a cut and its complement give the same row.
        cuts = tuple(
            (tuple(i for i in vertices if i not in subset), tuple(sorted(subset)))
            for count in range(1, size) for values in combinations(vertices[1:], count)
            for subset in (frozenset(values),)
        )
    else:
        cuts = tuple((tuple(i for i in vertices if i != v), (v,)) for v in vertices)
    connected = _connected(size, edges)
    if not connected:
        components = _Components(size)
        for edge in edges:
            components.join(*edge)
        side = tuple(v for v in vertices if components.root(v) == components.root(0))
        rest = tuple(v for v in vertices if v not in side)
        cuts = (*cuts, (side, rest))
    cuts = tuple(dict.fromkeys(connector_row(edges, groups) for groups in cuts))
    if full_partitions:
        additions = tuple(groups for groups in set_partitions(size) if len(groups) >= 3)
    else:
        singleton = tuple((i,) for i in vertices)
        additions = (singleton,) if illustration is None else (illustration, singleton)
    return ConnectorModels(graph, edges, cuts, tuple(connector_row(edges, p) for p in additions),
                           full_cuts, full_partitions, connected)


def connector_program(models: ConnectorModels, rows: tuple[ConnectorRow, ...]) -> LinearProgram:
    coefficients = tuple(models.graph.weight(edge) for edge in models.edges)
    size = len(coefficients)
    cuts = tuple(LinearConstraint(tuple(Fraction(i in row.indices) for i in range(size)),
                                  ConstraintRelation.GREATER_EQUAL, row.rhs) for row in rows)
    upper = tuple(LinearConstraint(tuple(Fraction(i == j) for i in range(size)),
                                   ConstraintRelation.LESS_EQUAL, 1) for j in range(size))
    return LinearProgram(ObjectiveSense.MINIMIZE, coefficients, (*cuts, *upper),
                         tuple(f"x{i + 1}" for i in range(size)), (VariableDomain.NONNEGATIVE,) * size)


def solve_connector_lp(models: ConnectorModels, *, strong: bool = False,
                       weak_bound: Fraction | None = None) -> ConnectorLP:
    """Solve the displayed relaxation, or return a certified bound on dense inputs."""

    if not models.connected:
        return ConnectorLP("infeasible")
    if not models.edges:
        return ConnectorLP("optimal", (), ZERO)
    rows = models.partition_rows if strong else models.cuts
    if len(models.edges) <= EXACT_LP_EDGES:
        problem = connector_program(models, rows)
        result = solve_simplex(problem, manual_start=(Fraction(1),) * len(models.edges))
        if result.status == "optimal":
            point = result.steps[-1].point
            return ConnectorLP("optimal", point, objective_value(problem, point))
    bound = _dual_bound(models, rows)
    if strong and weak_bound is not None:
        bound = max(bound, weak_bound)
    return ConnectorLP("bound_only", bound=bound)


def _dual_bound(models: ConnectorModels, rows: tuple[ConnectorRow, ...]) -> Fraction:
    """Build a feasible LP dual certificate with nonnegative row multipliers."""

    residual = [models.graph.weight(edge) for edge in models.edges]
    bound = sum((min(ZERO, cost) for cost in residual), ZERO)
    for row in rows:
        if not row.indices:
            continue
        multiplier = min(residual[i] for i in row.indices)
        if multiplier > 0:
            bound += multiplier * row.rhs
            for i in row.indices:
                residual[i] -= multiplier
    return bound


def choose_partition(models: ConnectorModels, weak: ConnectorLP) -> ConnectorRow | None:
    if not models.additions:
        return None
    if weak.point is not None:
        candidate = max(models.additions, key=lambda row: row.rhs - row.activity(weak.point))
        if candidate.activity(weak.point) < candidate.rhs:
            return candidate
    return models.additions[0]


def verify_integer_optimum(models: ConnectorModels) -> ConnectorOptima | None:
    """Compare the binary connector optimum with an independent Kruskal tree."""

    if not models.connected:
        return None
    graph = models.graph
    kruskal = minimum_spanning(graph, Method.KRUSKAL, 0)[-1]
    assert isinstance(kruskal.state, SpanningState)
    if len(models.edges) <= ENUMERATION_EDGES:
        connector, cost = _enumerate_connectors(graph, models.edges)
        method = "Exhaustive binary enumeration with a connectivity check"
    else:
        prim = minimum_spanning(graph, Method.PRIM, 0)[-1]
        # Prim's edge choices remain optimal with c replaced by max(c, 0).
        # All negative edges can then be added to a connector at zero extra positive cost.
        connector = prim.selected | frozenset(e for e in models.edges if graph.weight(e) < 0)
        cost = sum((graph.weight(edge) for edge in connector), ZERO)
        method = "Prim with nonpositive-edge handling, checked against Kruskal for the tree"
    return ConnectorOptima(connector, cost, kruskal.selected, kruskal.state.cost, method)


def _enumerate_connectors(graph: Graph, edges: tuple[Edge, ...]) -> tuple[frozenset[Edge], Fraction]:
    best = None
    for choices in product((False, True), repeat=len(edges)):
        if sum(choices) < graph.size - 1:
            continue
        subset = frozenset(edge for edge, selected in zip(edges, choices) if selected)
        value = sum((graph.weight(edge) for edge in subset), ZERO)
        key = (value, len(subset))
        if (best is None or key < best[0]) and _connected(graph.size, subset):
            best = key, subset
    assert best is not None
    return best[1], best[0][0]


@dataclass(frozen=True)
class Polytope:
    vertices: tuple[Vector, ...]
    faces: tuple[tuple[Vector, ...], ...]


def polytope_three(models: ConnectorModels, *, strong: bool = False) -> Polytope:
    """Intersect triples of boundary planes, keeping each feasible vertex and facet."""

    if len(models.edges) != 3 or not models.connected:
        return Polytope((), ())
    program = connector_program(models, models.partition_rows if strong else models.cuts)
    rows = inequalities(program)
    vertices = set()
    for triple in combinations(rows, 3):
        matrix = Matrix([row.coefficients for row in triple])
        if not matrix.det():
            continue
        point = tuple(Fraction(value) for value in matrix.inv() * Matrix([row.bound for row in triple]))
        if all(dot(row.coefficients, point) <= row.bound for row in rows):
            vertices.add(point)
    ordered = tuple(sorted(vertices))
    faces = tuple(dict.fromkeys(
        face for row in rows
        if len(face := tuple(p for p in ordered if dot(row.coefficients, p) == row.bound)) >= 3
    ))
    return Polytope(ordered, faces)


TRIANGLE = ((0, 1, 4), (0, 2, 7), (1, 2, 5))
# The slide gives the topology but no weights for the nine-node graph.
LECTURE_GRAPH = ((0, 1, 3), (0, 2, 4), (0, 3, 8), (0, 4, 5), (1, 4, 2),
                 (1, 7, 7), (2, 3, 6), (2, 5, 3), (3, 8, 2), (4, 6, 8),
                 (4, 7, 4), (5, 6, 7), (5, 8, 6), (6, 8, 4), (7, 8, 3))
LECTURE_PARTITION = ((0, 2, 3, 5), (1, 4, 7), (6, 8))
