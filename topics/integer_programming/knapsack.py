"""Exact fractional knapsack and reversible 0–1 branch-and-bound events."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction

from topics.integer_programming.models import NodeStatus
from topics.linear_programming.models import _as_fraction
from topics.simplex_method.solver import Vector


ZERO = Fraction(0)
ONE = Fraction(1)


class SearchOrder(str, Enum):
    DEPTH_FIRST = "Depth-First Search"
    BEST_BOUND = "Best-Bound"


@dataclass(frozen=True)
class Item:
    name: str
    weight: Fraction
    value: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", _as_fraction(self.weight))
        object.__setattr__(self, "value", _as_fraction(self.value))
        if not self.name.strip() or self.weight <= 0:
            raise ValueError("Each item needs a name and a strictly positive weight.")

    @property
    def density(self) -> Fraction:
        return self.value / self.weight


@dataclass(frozen=True)
class Knapsack:
    items: tuple[Item, ...]
    capacity: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity", _as_fraction(self.capacity))
        if self.capacity < 0 or not self.items:
            raise ValueError("Enter at least one item and a nonnegative capacity.")


@dataclass(frozen=True)
class Packing:
    x: Vector
    weight: Fraction
    value: Fraction


@dataclass(frozen=True)
class Relaxation:
    fixed_weight: Fraction
    lp: Packing | None
    heuristic: Packing | None
    fractional: int | None = None
    prefix: tuple[int, ...] = ()


Fixings = tuple[tuple[int, int], ...]


def density_order(problem: Knapsack) -> tuple[int, ...]:
    """Break equal ratios by input position. All solution vectors use that original order."""

    return tuple(sorted(range(len(problem.items)), key=lambda j: (-problem.items[j].density, j)))


def packing(problem: Knapsack, point: Iterable[Fraction]) -> Packing:
    x = tuple(point)
    if len(x) != len(problem.items):
        raise ValueError("A packing requires one coordinate per item.")
    return Packing(x, sum((a * item.weight for a, item in zip(x, problem.items)), ZERO),
                   sum((a * item.value for a, item in zip(x, problem.items)), ZERO))


def greedy_relaxation(problem: Knapsack, fixed: Fixings = (), *,
                       order: tuple[int, ...] | None = None) -> Relaxation:
    """Solve the LP with fixed variables and round the single fractional entry down."""

    values = [ZERO] * len(problem.items)
    fixed_indices = set()
    for index, value in fixed:
        if not 0 <= index < len(values) or index in fixed_indices or value not in (0, 1):
            raise ValueError("Fixings must specify distinct item indices and binary values.")
        fixed_indices.add(index)
        values[index] = Fraction(value)
    fixed_weight = sum((values[i] * problem.items[i].weight for i in fixed_indices), ZERO)
    remaining = problem.capacity - fixed_weight
    if remaining < 0:
        return Relaxation(fixed_weight, None, None)
    fractional = None
    prefix = []
    for index in order if order is not None else density_order(problem):
        if index in fixed_indices:
            continue
        item = problem.items[index]
        if item.value <= 0 or remaining == 0:
            break
        quantity = min(ONE, remaining / item.weight)
        values[index] = quantity
        prefix.append(index)
        remaining -= quantity * item.weight
        if quantity < 1:
            fractional = index
            break
    relaxed = packing(problem, values)
    if fractional is not None:
        values[fractional] = ZERO
    return Relaxation(fixed_weight, relaxed, packing(problem, values), fractional, tuple(prefix))


@dataclass(frozen=True)
class SearchNode:
    id: int
    parent: int | None
    fixed: Fixings = ()
    status: NodeStatus = NodeStatus.OPEN
    relaxation: Relaxation | None = None
    heuristic_known: bool = False
    order: int = 0
    reason: str = "Open subproblem"


@dataclass(frozen=True)
class SearchEvent:
    phase: str
    focus: int
    changes: tuple[SearchNode, ...]
    upper: Fraction | None = None
    incumbent: Packing | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class KnapsackRun:
    events: tuple[SearchEvent, ...]
    order: SearchOrder

    def nodes_at(self, index: int) -> dict[int, SearchNode]:
        nodes = {}
        for event in self.events[:index + 1]:
            nodes.update((node.id, node) for node in event.changes)
        return nodes


@dataclass(frozen=True)
class _OpenNode:
    node: SearchNode
    bound: Fraction | None
    relaxed: Relaxation | None = None


def knapsack_branch_and_bound(
    problem: Knapsack, strategy: SearchOrder = SearchOrder.DEPTH_FIRST, *,
    node_limit: int = 200, tolerance: Fraction = ZERO,
) -> KnapsackRun:
    """Record relaxed subproblems, incumbent updates and each branching or pruning decision."""

    tolerance = _as_fraction(tolerance)
    if node_limit < 1 or tolerance < 0:
        raise ValueError("Use a positive node limit and a nonnegative objective tolerance.")
    root = SearchNode(1, None)
    pending = [_OpenNode(root, None)]
    events = [SearchEvent("root", root.id, (root,))]
    order = density_order(problem)
    next_id = 2
    processed = 0
    incumbent = None
    outcome = "optimal"
    active_upper = None

    def publish(phase: str, node: SearchNode, *changes: SearchNode) -> None:
        events.append(SearchEvent(phase, node.id, (node, *changes), active_upper, incumbent))

    while pending and processed < node_limit:
        if strategy is SearchOrder.BEST_BOUND:
            # A cached infeasible LP comes last. Equal bounds follow creation order.
            selected = max(range(len(pending)), key=lambda i: (
                pending[i].bound is not None, pending[i].bound or ZERO, -pending[i].node.id,
            ))
            entry = pending.pop(selected)
        else:
            entry = pending.pop()
        node = entry.node
        relaxed = entry.relaxed or greedy_relaxation(problem, node.fixed, order=order)
        processed += 1
        active_upper = _global_upper(incumbent, pending, relaxed)
        node = replace(node, relaxation=relaxed, order=processed, status=NodeStatus.LP_SOLVED, reason="LP solved")
        publish("relaxation", node)
        if relaxed.lp is None:
            node = replace(node, status=NodeStatus.INFEASIBLE, reason="Infeasibility · Fixed weight exceeds capacity.")
        else:
            candidate = relaxed.heuristic
            assert candidate is not None
            improved = incumbent is None or candidate.value > incumbent.value
            if improved:
                incumbent = candidate
            node = replace(node, heuristic_known=True)
            publish("incumbent_updated" if improved else "incumbent_checked", node)
            if relaxed.fractional is None:
                node = replace(node, status=NodeStatus.INTEGER, reason="Optimality · The LP solution is binary.")
            elif relaxed.lp.value <= incumbent.value:
                node = replace(node, status=NodeStatus.PRUNED, reason="Dominance · Node UB ≤ global LB.")
            else:
                variable = relaxed.fractional
                children = tuple(SearchNode(next_id + i, node.id, (*node.fixed, (variable, value)))
                                 for i, value in enumerate((1, 0)))
                next_id += 2
                node = replace(node, status=NodeStatus.FRACTIONAL,
                               reason=f"Branch on x{variable + 1} = {relaxed.lp.x[variable]}.")
                for child in reversed(children):
                    if strategy is SearchOrder.BEST_BOUND:
                        cached_child = greedy_relaxation(problem, child.fixed, order=order)
                        bound = cached_child.lp.value if cached_child.lp else None
                        pending.append(_OpenNode(child, bound, cached_child))
                    else:
                        pending.append(_OpenNode(child, relaxed.lp.value))
                active_upper = _global_upper(incumbent, pending)
                publish("branched", node, *children)
        if node.status is not NodeStatus.FRACTIONAL:
            active_upper = _global_upper(incumbent, pending)
            publish("pruned", node)
        if tolerance and pending and incumbent and incumbent.value < active_upper <= incumbent.value + tolerance:
            outcome = "tolerance"
            break

    active_upper = _global_upper(incumbent, pending)
    if pending and outcome != "tolerance":
        # A previously computed upper bound can still certify optimality at the limit.
        outcome = "optimal" if incumbent and active_upper == incumbent.value else "limit"
    events.append(SearchEvent("finished", events[-1].focus, (), active_upper, incumbent, outcome))
    return KnapsackRun(tuple(events), strategy)


def _global_upper(
    incumbent: Packing | None, pending: list[_OpenNode],
    active: Relaxation | None = None,
) -> Fraction | None:
    bounds = [entry.bound for entry in pending if entry.bound is not None]
    if active and active.lp:
        bounds.append(active.lp.value)
    if incumbent:
        bounds.append(incumbent.value)
    return max(bounds, default=None)


LECTURE = Knapsack(tuple(Item(str(j), weight, value) for j, (weight, value) in enumerate(
    zip((5, 6, 12, 10, 12), (10, 9, 12, 5, 9)), 1,
)), 25)
