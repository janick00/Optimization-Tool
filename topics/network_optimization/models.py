"""Exact graph data and immutable algorithm snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


Weight = Fraction | None
Edge = tuple[int, int]
Matrix = tuple[tuple[Weight, ...], ...]


class Method(str, Enum):
    DFS = "Depth-First Search (DFS)"
    BFS = "Breadth-First Search (BFS)"
    KRUSKAL = "Optimistic Approach / Kruskal"
    REVERSE_DELETE = "Pessimistic Approach (Reverse-Delete)"
    PRIM = "Prim"
    DIJKSTRA = "Dijkstra"
    FLOYD = "Floyd-Warshall"

    @property
    def requires_undirected(self) -> bool:
        return self in (Method.KRUSKAL, Method.REVERSE_DELETE, Method.PRIM)

    @property
    def uses_start(self) -> bool:
        return self not in (Method.KRUSKAL, Method.REVERSE_DELETE)

    @property
    def uses_target(self) -> bool:
        return not self.requires_undirected


@dataclass(frozen=True)
class Graph:
    names: tuple[str, ...]
    weights: Matrix
    directed: bool = False

    def __post_init__(self) -> None:
        n = len(self.names)
        if not n or any(not name.strip() for name in self.names):
            raise ValueError("Jeder Knoten braucht einen Namen.")
        if len(set(self.names)) != n:
            raise ValueError("Knotennamen müssen eindeutig sein.")
        if len(self.weights) != n or any(len(row) != n for row in self.weights):
            raise ValueError("Die Adjazenzmatrix muss quadratisch sein.")
        if any(self.weights[i][i] != 0 for i in range(n)):
            raise ValueError("Die Diagonale muss 0 sein. Schleifen sind in dieser Maske ausgeschlossen.")
        if not self.directed and any(
            self.weights[i][j] != self.weights[j][i] for i in range(n) for j in range(i)
        ):
            raise ValueError("Ein ungerichteter Graph braucht eine symmetrische Matrix.")

    @property
    def size(self) -> int:
        return len(self.names)

    def edges(self) -> tuple[Edge, ...]:
        return tuple(
            (u, v) for u in range(self.size) for v, weight in enumerate(self.weights[u])
            if weight is not None and u != v and (self.directed or u < v)
        )

    def neighbors(self, vertex: int) -> tuple[int, ...]:
        return tuple(i for i, weight in enumerate(self.weights[vertex]) if i != vertex and weight is not None)

    def edge(self, u: int, v: int) -> Edge:
        return (u, v) if self.directed or u < v else (v, u)

    def weight(self, edge: Edge) -> Fraction:
        value = self.weights[edge[0]][edge[1]]
        if value is None:
            raise ValueError("Diese Kante existiert nicht.")
        return value

    def edge_name(self, edge: Edge) -> str:
        arrow = "→" if self.directed else "–"
        return f"{self.names[edge[0]]} {arrow} {self.names[edge[1]]}"


def is_missing(text: str) -> bool:
    return text.strip().lower() in ("", "∞", "inf", "+inf", "+∞")


def parse_graph(names: tuple[str, ...], cells: tuple[tuple[str, ...], ...], directed: bool) -> Graph:
    clean_names = tuple(name.strip() for name in names)
    rows = []
    for i, row in enumerate(cells):
        values = []
        for j, text in enumerate(row):
            if is_missing(text):
                values.append(Fraction(0) if i == j else None)
                continue
            try:
                values.append(Fraction(text.strip().replace(",", ".")))
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError(f"Ungültiges Gewicht in Zeile {i + 1}, Spalte {j + 1}: {text!r}.") from error
        rows.append(tuple(values))
    return Graph(clean_names, tuple(rows), directed)


@dataclass(frozen=True)
class TraversalState:
    """Keep the original graph highlights alongside both chronological orders."""

    order: tuple[int, ...]
    discovered: tuple[int, ...]
    frontier: tuple[int, ...]
    frontier_label: str
    parents: tuple[int | None, ...]
    discovery_order: tuple[int, ...]
    finishing_order: tuple[int, ...]
    depth_first: bool
    path: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SpanningState:
    cost: Fraction
    components: int
    remaining_label: str = "Gewählte Kanten"


@dataclass(frozen=True)
class DijkstraState:
    distances: tuple[Weight, ...]
    parents: tuple[int | None, ...]
    fixed: tuple[int, ...]
    start: int
    target: int | None
    final: bool = False


@dataclass(frozen=True)
class MatrixChange:
    source: int
    target: int
    before: Weight
    after: Fraction


@dataclass(frozen=True)
class FloydState:
    distances: Matrix
    via: int | None = None
    changes: tuple[MatrixChange, ...] = ()
    unbounded: frozenset[Edge] = frozenset()
    path: tuple[int, ...] | None = None
    query: Edge | None = None


@dataclass(frozen=True)
class Step:
    title: str
    explanation: str
    state: TraversalState | SpanningState | DijkstraState | FloydState
    selected: frozenset[Edge] = frozenset()
    rejected: frozenset[Edge] = frozenset()
    focus_nodes: tuple[int, ...] = ()
    focus_edge: Edge | None = None
    path: tuple[Edge, ...] = ()


def predecessor_path(parents: tuple[int | None, ...] | list[int | None],
                     start: int, target: int) -> tuple[int, ...] | None:
    path = [target]
    while path[-1] != start:
        previous = parents[path[-1]]
        if previous is None or len(path) == len(parents):
            return None
        path.append(previous)
    return tuple(reversed(path))


def path_edges(graph: Graph, path: tuple[int, ...] | None) -> tuple[Edge, ...]:
    return tuple(graph.edge(u, v) for u, v in zip(path, path[1:])) if path else ()
