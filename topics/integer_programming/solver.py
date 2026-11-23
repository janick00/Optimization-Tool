"""Exact Branch and Bound over the existing simplex relaxations."""

from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from math import ceil, floor

from topics.integer_programming.feasibility import check_integer_point
from topics.integer_programming.models import (
    Bounds, Branch, Domain, Event, EventKind, Incumbent, Node, NodeStatus, TreeRun,
)
from topics.linear_programming.models import (
    ConstraintRelation, LinearConstraint, LinearProgram, ObjectiveSense, _as_fraction,
)
from topics.simplex_method.solver import objective_value, solve_simplex

DEFAULT_NODE_LIMIT = 100


def relaxation(problem: LinearProgram, domains: tuple[Domain, ...]) -> LinearProgram:
    """Keep the original sign constraints and include both bounds for binaries."""

    if len(domains) != len(problem.objective):
        raise ValueError("Choose a domain for every variable.")
    binary = tuple(
        _branch_row(problem, Branch(i, relation, bound), "Binary")
        for i, domain in enumerate(domains) if domain is Domain.BINARY
        for relation, bound in ((ConstraintRelation.GREATER_EQUAL, 0), (ConstraintRelation.LESS_EQUAL, 1))
    )
    return replace(problem, constraints=problem.constraints + binary)


def _branch_row(problem: LinearProgram, branch: Branch, prefix: str = "Branch") -> LinearConstraint:
    row = tuple(Fraction(i == branch.variable) for i in range(len(problem.objective)))
    return LinearConstraint(row, branch.relation, Fraction(branch.bound),
                            f"{prefix}: {branch.label(problem.variable_names)}")


def branch_and_bound(
    problem: LinearProgram, domains: tuple[Domain, ...], *, node_limit: int = DEFAULT_NODE_LIMIT,
    initial_incumbent: Sequence[Fraction | int | float | str] | None = None,
) -> TreeRun:
    """Solve breadth-first, branching on the first fractional discrete variable."""

    if node_limit < 1:
        raise ValueError("Node limit must be at least 1.")
    root_problem = relaxation(problem, domains)
    seed = _validate_incumbent(problem, domains, initial_incumbent) if initial_incumbent is not None else None
    maximize = problem.objective_sense is ObjectiveSense.MAXIMIZE
    best = max if maximize else min
    root = Node(1, None, (), root_problem, 0)
    nodes = {1: root}
    pending = deque([1])
    frontier: dict[int, Fraction | None] = {1: None}
    incumbent: Incumbent | None = None
    events: list[Event] = []
    processed = 0

    def publish(kind: EventKind, text: str, focus: int, *changed: Node, outcome: str | None = None) -> None:
        nodes.update((node.id, node) for node in changed)
        primal = incumbent.value if incumbent else None
        finite = [bound for bound in frontier.values() if bound is not None]
        if primal is not None:
            finite.append(primal)
        dual = None if None in frontier.values() else best(finite, default=None)
        bounds = Bounds(primal, dual) if maximize else Bounds(dual, primal)
        events.append(Event(kind, text, focus, changed, incumbent, bounds, outcome))

    publish(EventKind.ROOT_CREATED, "Root node created.", 1, root)
    if seed is not None:
        incumbent = seed
        publish(EventKind.INITIAL_INCUMBENT, "Feasible x₀ accepted as the initial incumbent.", 1)
    while pending and processed < node_limit:
        node_id = pending.popleft()
        node = nodes[node_id]
        processed += 1
        simplex = solve_simplex(node.problem, capture_details=True)
        node = replace(
            node, simplex=simplex, processed_order=processed, status=NodeStatus.LP_SOLVED,
            reason=simplex.message,
        )
        if simplex.status == "infeasible":
            publish(EventKind.LP_SOLVED, f"Node {node_id}: LP relaxation is infeasible.", node_id, node)
            frontier.pop(node_id)
            node = replace(node, status=NodeStatus.INFEASIBLE, reason="Pruned because its LP relaxation is infeasible.")
            publish(EventKind.PRUNED, f"Node {node_id}: pruned by infeasibility.", node_id, node)
            continue
        if simplex.status != "optimal":
            node = replace(node, status=NodeStatus.UNRESOLVED)
            publish(EventKind.LP_SOLVED, f"Node {node_id}: relaxation unresolved.", node_id, node)
            continue

        point = simplex.steps[-1].point
        value = objective_value(node.problem, point)
        node = replace(node, solution=point, value=value, reason="LP relaxation solved. Integrality check pending.")
        publish(EventKind.LP_SOLVED, f"Node {node_id}: LP relaxation solved.", node_id, node)

        fractional = tuple(i for i, x in enumerate(point) if domains[i] is not Domain.CONTINUOUS and x.denominator != 1)
        node = replace(
            node, fractional=fractional, status=NodeStatus.FRACTIONAL if fractional else NodeStatus.INTEGER,
            reason="At least one discrete variable is fractional." if fractional
            else "All discrete variables satisfy their domains.",
        )
        publish(EventKind.INTEGRALITY_CHECKED, f"Node {node_id}: integrality checked.", node_id, node)

        frontier[node_id] = value
        local = (Bounds(upper=value) if maximize else Bounds(lower=value)) if fractional else Bounds(value, value)
        node = replace(node, bounds=local)
        dominated = incumbent is not None and best(value, incumbent.value) == incumbent.value
        update = f"Node {node_id}: bounds updated."
        if not fractional and not dominated:
            frontier.pop(node_id)
            incumbent = Incumbent(node_id, point, value)
            update = f"Node {node_id}: bounds updated with new incumbent J = {value}."
        publish(EventKind.BOUNDS_UPDATED, update, node_id, node)
        if dominated:
            frontier.pop(node_id)
            node = replace(
                node, status=NodeStatus.PRUNED,
                reason=f"LP bound {value} cannot improve the incumbent {incumbent.value}.",
            )
            publish(EventKind.PRUNED, f"Node {node_id}: pruned by bound.", node_id, node)
        elif fractional:
            index = fractional[0]
            node = replace(
                node, status=NodeStatus.FRACTIONAL, branch_variable=index,
                reason=f"{problem.variable_names[index]} = {point[index]} is the first fractional discrete value.",
            )
            publish(EventKind.BRANCH_SELECTED, f"Node {node_id}: branching variable selected.", node_id, node)
            children = []
            for relation, bound in (
                (ConstraintRelation.LESS_EQUAL, floor(point[index])),
                (ConstraintRelation.GREATER_EQUAL, ceil(point[index])),
            ):
                branch = Branch(index, relation, bound)
                child_id = len(nodes) + 1 + len(children)
                child_problem = replace(
                    node.problem, constraints=node.problem.constraints + (_branch_row(problem, branch),)
                )
                children.append(Node(child_id, node_id, (*node.branches, branch), child_problem,
                                     len(events), bounds=local))
                pending.append(child_id)
                frontier[child_id] = value
            frontier.pop(node_id)
            publish(EventKind.CHILDREN_CREATED, f"Node {node_id}: created children {children[0].id} and {children[1].id}.",
                    node_id, *children)

    if pending:
        publish(EventKind.FINISHED, f"Node limit reached after {processed} solved LPs. Search is incomplete.",
                pending[0], outcome="limit")
        return TreeRun(tuple(events), "limit")
    if frontier:
        publish(EventKind.FINISHED, "Search incomplete. At least one LP relaxation was unresolved.",
                next(iter(frontier)), outcome="unresolved")
        return TreeRun(tuple(events), "unresolved")
    if incumbent is None:
        publish(EventKind.FINISHED, "Search complete. No feasible integer solution exists.",
                events[-1].focus, outcome="infeasible")
        return TreeRun(tuple(events), "infeasible")
    if incumbent.node is None:
        publish(EventKind.FINISHED, "Global optimum proven for the initial incumbent x₀.",
                events[-1].focus, outcome="optimal")
    else:
        winning = replace(nodes[incumbent.node], status=NodeStatus.OPTIMAL,
                          reason="All remaining branches have been ruled out. Global optimality is proven.")
        publish(EventKind.FINISHED, f"Global optimum proven at node {winning.id}.",
                winning.id, winning, outcome="optimal")
    return TreeRun(tuple(events), "optimal")


def _validate_incumbent(
    problem: LinearProgram, domains: tuple[Domain, ...], values: Sequence[Fraction | int | float | str],
) -> Incumbent:
    point = tuple(_as_fraction(value) for value in values)
    if len(point) != len(domains):
        raise ValueError(f"Initial solution x₀ requires exactly {len(domains)} coordinates.")
    check = check_integer_point(problem, domains, point)
    if check.value is None:
        issues = (*check.integrality_errors, *check.domain_errors, *check.constraint_errors)
        raise ValueError(f"Initial solution x₀ is infeasible: {' '.join(issues)}")
    return Incumbent(None, point, check.value)
