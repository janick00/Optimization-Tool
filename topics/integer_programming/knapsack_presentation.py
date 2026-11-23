"""Calculation text and capacity diagram for the knapsack lesson."""

from html import escape

from topics.integer_programming.formatting import _document, point_text
from topics.integer_programming.knapsack import Knapsack, KnapsackRun, Packing, SearchNode, SearchOrder, packing


COLORS = ("#60a5fa", "#2dd4bf", "#fbbf24", "#c084fc", "#fb7185", "#a3e635")
FORMULA = "max z = Σ cⱼxⱼ · Σ aⱼxⱼ ≤ b · xⱼ ∈ {0, 1}"


def fixing_text(node: SearchNode) -> str:
    return ", ".join(f"x{i + 1} = {value}" for i, value in node.fixed) or "∅"


def node_lines(node: SearchNode) -> tuple[str, ...]:
    lp = node.relaxation.lp if node.relaxation else None
    local = node.relaxation.heuristic if node.relaxation and node.heuristic_known else None
    return (
        f"{'Root' if node.parent is None else 'Node'} {node.id}"
        + (f" · Visit {node.order}" if node.order else ""),
        f"Fixings {fixing_text(node)}",
        f"x LP = {point_text(lp.x if lp else None)}",
        f"UB {lp.value if lp else '—'} · LB {local.value if local else '—'}",
        node.status.value, node.reason,
    )


def branch_label(node: SearchNode) -> str:
    i, value = node.fixed[-1]
    return f"x{i + 1} = {value}"


def plot_capacity(figure, problem: Knapsack | None, current: Packing | None, order=()) -> None:
    figure.clear()
    axis = figure.add_subplot()
    axis.set_facecolor("#111827")
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(-.65, .65)
    figure.subplots_adjust(left=.05, right=.95, bottom=.10, top=.90)
    if problem is None:
        axis.text(.5, 0, "Capacity preview", ha="center", va="center", color="#cbd5e1")
        return
    axis.barh(0, 1, height=.42, color="#334155", edgecolor="#64748b", zorder=1)
    if current and problem.capacity:
        left = 0.
        for i in order:
            portion = current.x[i]
            if not portion:
                continue
            width = float(portion * problem.items[i].weight / problem.capacity)
            axis.barh(0, width, left=left, height=.42, color=COLORS[i % len(COLORS)],
                      edgecolor="#e2e8f0", linewidth=.7, hatch="///" if portion < 1 else None, zorder=2)
            if width >= .055:
                axis.text(left + width / 2, 0, f"x{i + 1}", ha="center", va="center",
                          color="#0f172a", fontsize=8, weight="bold", zorder=3)
            left += width
    used = current.weight if current else 0
    axis.text(0, .43, f"Capacity {problem.capacity}", color="#e2e8f0", fontsize=9, va="center",
              transform=axis.transData)
    axis.text(0, -.42, f"Used {used} · Remaining {problem.capacity - used}",
              color="#cbd5e1", fontsize=8, va="center")


def greedy_frame(problem, order, relaxed, index) -> tuple[str, Packing | None]:
    lp, heuristic = relaxed.lp, relaxed.heuristic
    assert lp is not None and heuristic is not None
    original = " → ".join(f"x{i + 1}" for i in range(len(order)))
    sorted_items = " → ".join(f"x{i + 1} ({problem.items[i].density})" for i in order)
    body = (f"<h3>0–1 Knapsack</h3><p>{FORMULA}<br/>Capacity b = {problem.capacity}. "
            f"The LP instead uses 0 ≤ xⱼ ≤ 1.</p><p><b>Input order:</b> {original}</p>")
    if index == 0:
        return _document(body + "<p>Calculate cⱼ/aⱼ for each row of the item table.</p>"), None
    body += f"<p><b>Sorted by cⱼ/aⱼ:</b> {sorted_items}</p>"
    if index == 1:
        return _document(body + "<p>Pack positive-value items in this order.</p>"), None
    used = index - 1
    if used <= len(relaxed.prefix):
        current_item = relaxed.prefix[used - 1]
        previous = set(relaxed.prefix[:used])
        point = packing(problem, (value if i in previous else 0 for i, value in enumerate(lp.x)))
        item = problem.items[current_item]
        choice = lp.x[current_item]
        explanation = ("The remaining capacity fits only a fraction of this item." if choice < 1 else
                       "This item fits completely.")
        return _document(body + (
            f"<h3>Pack x{current_item + 1} = {choice}</h3><p>{escape(item.name)} has weight {item.weight} "
            f"and value {item.value}. {explanation}</p>{_packing_text(problem, point)}"
        )), point
    if used == len(relaxed.prefix) + 1:
        sorted_point = tuple(lp.x[i] for i in order)
        return _document(body + (
            f"<h3>LP optimum</h3><p>x in sorted order = {point_text(sorted_point)}.</p>"
            f"{_packing_text(problem, lp)}<p><b>Upper bound UB = {lp.value}.</b></p>"
        )), lp
    return _document(body + (
        "<h3>Feasible binary solution</h3><p>Keep the full items and set any fractional variable to zero.</p>"
        f"{_packing_text(problem, heuristic)}<p><b>LB = {heuristic.value} · UB = {lp.value}.</b> "
        "The binary packing is a heuristic incumbent. The bounds certify optimality when they coincide.</p>"
    )), heuristic


def _packing_text(problem: Knapsack, point: Packing) -> str:
    value = " + ".join(f"{portion}·{item.value}" for portion, item in zip(point.x, problem.items) if portion)
    value = value or "0"
    return (f"<p>x in input order = {point_text(point.x)}<br/>"
            f"Weight = {point.weight}, remaining = {problem.capacity - point.weight}<br/>"
            f"z = {value} = {point.value}</p>")


def search_frame(problem: Knapsack, run: KnapsackRun, index: int, nodes, selected, order) -> str:
    event = run.events[index]
    node = nodes[selected]
    relaxed = node.relaxation
    body = (
        f"<h3>{run.order.value}</h3><p>{FORMULA}<br/>"
        f"Inspecting {'root' if node.parent is None else 'node'} {node.id}. Fixings {fixing_text(node)}.</p>"
    )
    if node.parent is None:
        body += "<p>Density order: " + " → ".join(f"x{i + 1}" for i in order) + ".</p>"
    if relaxed:
        body += f"<p>Remaining before Greedy: {problem.capacity} − {relaxed.fixed_weight} = "
        body += f"{problem.capacity - relaxed.fixed_weight}.</p>"
        if relaxed.lp:
            sorted_x = point_text(tuple(relaxed.lp.x[i] for i in order))
            body += (f"<h3>Node LP · UB = {relaxed.lp.value}</h3>{_packing_text(problem, relaxed.lp)}"
                     f"<p>x in density order = {sorted_x}</p>")
        if relaxed.heuristic and node.heuristic_known:
            body += (f"<p>Rounded binary point {point_text(relaxed.heuristic.x)} "
                     f"gives node LB = {relaxed.heuristic.value}.</p>")
    body += f"<p><b>{escape(node.reason)}</b></p>"
    lower = event.incumbent.value if event.incumbent else "—"
    upper = event.upper if event.upper is not None else "—"
    if event.phase == "incumbent_updated":
        body += "<p>The rounded solution improves the global incumbent.</p>"
    elif event.phase == "branched":
        note = ("Open children are ranked by their relaxed upper bound." if run.order is SearchOrder.BEST_BOUND
                else "Depth-first visits the child with xⱼ = 1 first. Open children inherit their parent's bound.")
        body += f"<p>{note}</p>"
    body += f"<p><b>Global LB {lower} · UB {upper}</b></p>"
    if event.outcome == "optimal" and event.incumbent:
        body += ("<h3>Optimal binary packing</h3><p>The global bounds coincide.</p>"
                 f"{_packing_text(problem, event.incumbent)}")
    elif event.outcome in ("limit", "tolerance"):
        why = "Node limit reached" if event.outcome == "limit" else "Objective tolerance reached"
        gap = event.upper - event.incumbent.value
        body += f"<p><b>{why}.</b> Remaining certified gap {gap}.</p>"
    return _document(body + _visits_table(nodes.values()))


def _visits_table(nodes) -> str:
    visits = sorted((node for node in nodes if node.order), key=lambda node: node.order)
    if not visits:
        return ""
    rows = []
    for node in visits:
        relaxed = node.relaxation
        assert relaxed is not None
        ub = relaxed.lp.value if relaxed.lp else "—"
        lb = relaxed.heuristic.value if relaxed.heuristic and node.heuristic_known else "—"
        decision = node.reason.split(" · ")[0].split(" on ")[0]
        rows.append(f"<tr><td>{node.order}</td><td>{node.id}</td><td>{ub}</td><td>{lb}</td>"
                    f"<td>{escape(decision)}</td></tr>")
    return ("<h3>Processing order</h3><p>k = visit, r = node</p>"
            "<table cellspacing='0' cellpadding='1'><tr><th>k</th><th>r</th><th>UB</th><th>LB</th><th>Action</th></tr>"
            + "".join(rows) + "</table>")
