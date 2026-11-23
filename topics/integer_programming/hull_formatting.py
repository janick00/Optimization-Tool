"""Calculation steps for the three hull and cutting-plane views."""

from html import escape

from topics.integer_programming.cutting_planes import Cut, CuttingSession
from topics.integer_programming.formatting import _document, point_text
from topics.integer_programming.hull_geometry import HullConstruction, dot
from topics.integer_programming.integer_hull import IntegerHull, RelaxationResult
from topics.linear_programming.models import ObjectiveSense


def _points(points) -> str:
    return ", ".join(map(point_text, points)) or "∅"


def _lp(lp: RelaxationResult | None) -> str:
    if lp is None:
        return "Pending"
    if lp.optimum is None:
        return lp.status.capitalize()
    return f"{point_text(lp.optimum.point)}, z = {lp.optimum.value}"


def _reference_status(result: IntegerHull) -> str:
    return {
        "complete": "",
        "empty_lp": "P is empty, so the integer-feasible set and its hull are empty.",
        "empty_discrete": "No feasible point satisfies the selected variable domains. The integer hull is empty.",
        "unbounded_region": "P is unbounded. Finite enumeration is unavailable and the integer hull is undetermined.",
        "search_truncated": "Search range truncated. The full discrete coordinate ranges exceed the selected search radius.",
        "point_limit": "Point limit reached. Increase the limit to enumerate the full feasible set.",
    }[result.status]


def format_construction(construction: HullConstruction, index: int) -> str:
    unique = len(set(construction.points))
    info = (
        "<h3>Convex hull</h3><p>conv(S) = {Σ λᵢpᵢ | λᵢ ≥ 0, Σ λᵢ = 1, pᵢ ∈ S}. "
        "It is the smallest convex set containing the input points.</p>"
        f"<p>{len(construction.points)} input points, {unique} distinct. "
        "The monotone chain builds the lower and upper boundaries in sorted order.</p>"
    )
    if index == 0:
        return _document(info + f"<p><b>Start.</b> Sort unique points: {_points(sorted(set(construction.points)))}</p>")
    if index > len(construction.steps):
        vertices = construction.vertices
        shape = {1: "a single point", 2: "a line segment"}.get(len(vertices), "a polygon")
        return _document(info + f"<p><b>Completed.</b> The hull is {shape} with vertices "
                         f"in boundary order: {_points(vertices)}.</p>")
    step = construction.steps[index - 1]
    point = point_text(step.point)
    if step.action == "consider":
        details = f"Consider {point} for the {step.chain.lower()} boundary."
    elif step.action == "remove":
        turn = "collinear" if step.turn == 0 else "a right turn"
        details = (f"Remove {point}. With candidate {point_text(step.candidate)}, "
                   f"the oriented determinant is {step.turn} ({turn}).")
    else:
        details = f"Add {point} to the {step.chain.lower()} chain."
    lower, upper = construction.chains_at(index)
    return _document(info + f"<h3>Step {index} · {step.chain} chain</h3><p>{details}</p>"
                     f"<p><b>Lower:</b> {_points(lower)}<br/><b>Upper:</b> {_points(upper)}</p>")


def format_integer_hull(result: IntegerHull, sense: ObjectiveSense) -> str:
    body = (
        "<h3>1. Continuous region P and LP relaxation</h3>"
        f"<p><b>Vertices of P:</b> {_points(result.region.vertices)}<br/>"
        f"<b>LP optimum:</b> {_lp(result.lp)}</p>"
        "<p>The LP drops integrality. A vertex of P can therefore have fractional coordinates.</p>"
        "<h3>2. Integer or mixed-integer hull</h3>"
        "<p>For two discrete variables, P<sub>I</sub> = conv(P ∩ ℤ²). "
        "With one continuous variable, each feasible integer coordinate can define an entire segment.</p>"
    )
    if result.status != "complete":
        return _document(body + f"<p>{_reference_status(result)}</p>")
    if result.segments:
        segments = ", ".join(f"[{point_text(a)}, {point_text(b)}]" for a, b in result.segments)
        body += f"<p><b>Feasible segments:</b> {segments}</p>"
    if result.points:
        body += f"<p><b>Feasible discrete points ({len(result.points)}):</b> {_points(result.points)}</p>"
    optimum, hull = result.discrete_optimum, result.hull_optimum
    assert optimum is not None and hull is not None
    return _document(body + (
        f"<p><b>Hull vertices:</b> {_points(result.vertices)}</p>"
        "<h3>3. Compare the optima</h3>"
        f"<p><b>ILP optimum:</b> {point_text(optimum.point)}, z = {optimum.value}<br/>"
        f"<b>Hull optimum:</b> {point_text(hull.point)}, z = {hull.value}<br/>"
        f"<b>Absolute integrality gap:</b> |z<sub>LP</sub> − z<sub>ILP</sub>| = {result.gap}</p>"
        f"<p>{sense.value} cᵀx over the feasible discrete set = {sense.value} cᵀx over its convex hull "
        f"= {hull.value}. On a convex combination, the linear objective is a weighted average of the "
        "original objective values. Every convex formulation containing the discrete feasible set also "
        "contains its hull, so the hull gives the tightest possible LP bound.</p>"
    ))


def _expression(coefficients, names) -> str:
    terms = []
    for coefficient, name in zip(coefficients, names):
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        term = (str(magnitude) + "·" if magnitude != 1 else "") + escape(name)
        sign = " − " if coefficient < 0 else " + "
        terms.append(sign + term if terms else ("−" if coefficient < 0 else "") + term)
    return "".join(terms) or "0"


def cut_text(cut: Cut, names) -> str:
    return f"{_expression(cut.coefficients, names)} ≤ {cut.bound}"


def format_cutting_step(session: CuttingSession) -> str:
    frame = session.frame
    names = frame.problem.variable_names
    body = "<h3>Cutting planes</h3>"
    if frame.phase == "region":
        body += "<p><b>Original LP relaxation.</b> The blue region is P. Next, solve the LP.</p>"
    elif frame.phase == "applied":
        body += ("<p><b>Cut added.</b> P becomes P ∩ {x | "
                 f"{cut_text(frame.cuts[-1], names)}}}. Re-solving the LP is the next step.</p>")
    elif frame.phase == "checked":
        if frame.fractional:
            assert frame.lp and frame.lp.optimum
            values = ", ".join(f"{escape(names[i])} = {frame.lp.optimum.point[i]}" for i in frame.fractional)
            body += f"<p><b>Integrality check:</b> {values} must be discrete. The optimum is fractional.</p>"
        else:
            body += "<p><b>Integrality check:</b> All variables satisfy their selected domains.</p>"
    elif frame.message:
        body += f"<p>{escape(frame.message)}</p>"
    if frame.lp is not None:
        body += f"<p><b>Current LP result:</b> {_lp(frame.lp)}.</p>"
    if frame.candidate:
        cut = frame.candidate
        assert frame.lp and frame.lp.optimum
        body += f"<h3>Cut {len(frame.cuts) + 1}</h3>"
        if cut.combination:
            weights = " + ".join(f"{u} · {escape(row.name)}" for u, row in cut.combination)
            rows = "<br/>".join(
                f"{u} · ({_expression(row.coefficients, names)} ≤ {row.bound})" for u, row in cut.combination
            )
            expression = _expression(cut.coefficients, names)
            body += (
                f"<p>Gomory–Chvátal: {weights}<br/>{rows}<br/>"
                f"⇒ {expression} ≤ {cut.combined_bound}</p>"
                f"<p>The left-hand side is integer on the feasible discrete set. "
                f"Round down: ⌊{cut.combined_bound}⌋ = {cut.bound}.</p>"
            )
        else:
            body += f"<p>{escape(cut.validation)}</p>"
        value = dot(cut.coefficients, frame.lp.optimum.point)
        body += (f"<p><b>{cut_text(cut, names)}</b><br/>At the current optimum the left-hand side is "
                 f"{value} &gt; {cut.bound}. The shaded region is removed when the cut is applied.</p>")
    if frame.comparisons:
        body += _comparison_table(session)
    return _document(body + (
        "<p>A valid cut shrinks the LP feasible set while preserving all points satisfying the integer domains. "
        "This can tighten the upper bound in maximization or the lower bound in minimization.</p>"
    ))


def _comparison_table(session: CuttingSession) -> str:
    frame = session.frame
    maximize = frame.problem.objective_sense is ObjectiveSense.MAXIMIZE
    side = "up" if maximize else "low"
    bound_label = "Upper bound" if maximize else "Lower bound"
    reference = session.reference
    optimum = reference.discrete_optimum
    proved_value = optimum.value if optimum else "—"
    if frame.phase == "finished" and frame.lp and frame.lp.optimum and not frame.fractional:
        proved_value = frame.lp.optimum.value
    lines = []
    for index, entry in enumerate(frame.comparisons):
        cut = str(index) if entry.cut else "—"
        bounds = entry.lp.optimum.value if entry.lp.optimum else "—"
        lp = entry.lp.optimum.value if entry.lp.optimum else entry.lp.status.capitalize()
        lines.append(f"<tr><td>{index}</td><td>{cut}</td><td>{lp}</td>"
                     f"<td>{bounds}</td><td>{proved_value}</td></tr>")
    if frame.phase == "applied":
        index = len(frame.cuts)
        lines.append(f"<tr><td>{index}</td><td>{index}</td><td>Pending</td>"
                     f"<td>—</td><td>{proved_value}</td></tr>")
    cuts = "".join(f"<li>{cut_text(cut, frame.problem.variable_names)}</li>" for cut in frame.cuts)
    return (f"<h3>Objective comparison</h3><p>Step k. {bound_label} z<sub>{side}</sub> comes from the LP.</p>"
            "<table cellspacing='1' cellpadding='1'><tr><th>k</th><th>Cut</th><th>z<sub>LP</sub></th>"
            f"<th>z<sub>{side}</sub></th><th>z<sub>ILP</sub></th></tr>{''.join(lines)}</table>"
            + (f"<p><b>Applied cuts</b></p><ol>{cuts}</ol>" if cuts else "")
            + (f"<p>{_reference_status(reference)}</p>" if proved_value == "—" else ""))
