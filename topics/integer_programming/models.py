"""Immutable solver records for chronological playback."""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from topics.linear_programming.models import ConstraintRelation, LinearProgram
from topics.simplex_method.solver import SimplexResult, Vector


class Domain(str, Enum):
    INTEGER = "Integer"
    BINARY = "Binary"
    CONTINUOUS = "Continuous"


class NodeStatus(str, Enum):
    OPEN = "Open"
    LP_SOLVED = "LP solved"
    FRACTIONAL = "Fractional"
    INTEGER = "Integer"
    PRUNED = "Pruned by Bound"
    INFEASIBLE = "Infeasible"
    OPTIMAL = "Optimal"
    UNRESOLVED = "Unresolved"


class EventKind(str, Enum):
    ROOT_CREATED = "root_created"
    INITIAL_INCUMBENT = "initial_incumbent"
    LP_SOLVED = "lp_solved"
    INTEGRALITY_CHECKED = "integrality_checked"
    BOUNDS_UPDATED = "bounds_updated"
    BRANCH_SELECTED = "branch_selected"
    CHILDREN_CREATED = "children_created"
    PRUNED = "pruned"
    FINISHED = "finished"


@dataclass(frozen=True)
class Branch:
    variable: int
    relation: ConstraintRelation
    bound: int

    def label(self, names: tuple[str, ...]) -> str:
        symbol = "≤" if self.relation is ConstraintRelation.LESS_EQUAL else "≥"
        return f"{names[self.variable]} {symbol} {self.bound}"


@dataclass(frozen=True)
class Bounds:
    lower: Fraction | None = None
    upper: Fraction | None = None


@dataclass(frozen=True)
class Node:
    id: int
    parent: int | None
    branches: tuple[Branch, ...]
    problem: LinearProgram
    created_at: int
    status: NodeStatus = NodeStatus.OPEN
    processed_order: int | None = None
    simplex: SimplexResult | None = None
    solution: Vector | None = None
    value: Fraction | None = None
    bounds: Bounds = Bounds()
    fractional: tuple[int, ...] | None = None
    branch_variable: int | None = None
    reason: str = "Waiting for the LP relaxation."


@dataclass(frozen=True)
class Incumbent:
    node: int | None
    point: Vector
    value: Fraction


@dataclass(frozen=True)
class Event:
    kind: EventKind
    text: str
    focus: int
    changes: tuple[Node, ...]
    incumbent: Incumbent | None
    bounds: Bounds
    outcome: str | None = None


@dataclass(frozen=True)
class TreeFrame:
    nodes: dict[int, Node]
    event: Event


@dataclass(frozen=True)
class TreeRun:
    events: tuple[Event, ...]
    status: str

    def frame(self, step: int) -> TreeFrame:
        if not 0 <= step < len(self.events):
            raise IndexError("Tree step is out of range.")
        nodes = {}
        for event in self.events[:step + 1]:
            nodes.update((node.id, node) for node in event.changes)
        return TreeFrame(nodes, self.events[step])
