"""Exact LP, bound and simplex data for the selected playback event."""

from html import escape
from fractions import Fraction

from topics.integer_programming.feasibility import PointCheck, relaxation_bounds, relaxation_value
from topics.integer_programming.models import Bounds, Domain, Node, TreeFrame
from topics.linear_programming.models import LinearProgram, ObjectiveSense
from topics.simplex_method.solver import SimplexResult


def point_text(point) -> str:
    return "—" if point is None else "(" + ", ".join(map(str, point)) + ")"


def bounds_text(bounds: Bounds) -> str:
    lower = "−∞" if bounds.lower is None else str(bounds.lower)
    upper = "+∞" if bounds.upper is None else str(bounds.upper)
    return f"Lower: {lower} · Upper: {upper}"


def _document(content: str) -> str:
    return (
        "<html><style>body{color:#dbeafe;font-size:10pt} h3,th,b{color:#93c5fd}"
        " td,th{padding:3px 5px;border-bottom:1px solid #334155}</style>"
        f"<body>{content}</body></html>"
    )


def _lp(problem: LinearProgram) -> str:
    names = problem.variable_names
    constraints = "".join(
        f"<tr><td>{escape(row.label or f'C{i}')}</td>"
        + "".join(f"<td>{value}</td>" for value in row.coefficients)
        + f"<td>{escape(row.relation.value)}</td><td>{row.right_hand_side}</td></tr>"
        for i, row in enumerate(problem.constraints, 1)
    )
    return (
        f"<h3>LP subproblem</h3><p>{problem.objective_sense.label} "
        f"J = {point_text(problem.objective)} · x</p>"
        "<table><tr><th>Constraint</th>"
        + "".join(f"<th>{escape(name)}</th>" for name in names)
        + f"<th>Relation</th><th>RHS</th></tr>{constraints}</table>"
        "<p>LP variable domains: "
        + escape(", ".join(f"{name}: {domain.value}" for name, domain in zip(names, problem.variable_domains)))
        + "</p>"
    )


def format_node(node: Node, frame: TreeFrame, domains: tuple[Domain, ...]) -> str:
    names = node.problem.variable_names
    branches = ", ".join(branch.label(names) for branch in node.branches) or "None"
    if node.solution is None:
        integrality = "Unavailable"
    elif node.fractional is None:
        integrality = "Pending"
    else:
        integrality = ", ".join(
            f"{name}: {value} ({'continuous' if domain is Domain.CONTINUOUS else 'fractional' if i in node.fractional else 'satisfied'})"
            for i, (name, domain, value) in enumerate(zip(names, domains, node.solution))
        )
    branching = "—" if node.branch_variable is None or node.solution is None else (
        f"{names[node.branch_variable]} = {node.solution[node.branch_variable]}"
    )
    event = frame.event
    incumbent = event.incumbent
    incumbent_info = "None"
    final = ""
    if incumbent is not None:
        source = "Initial solution x₀" if incumbent.node is None else f"Node {incumbent.node}"
        incumbent_info = f"{source} · {point_text(incumbent.point)} · J = {incumbent.value}"
        if event.outcome == "optimal":
            final = f"<p><b>Global optimum:</b> x = {point_text(incumbent.point)} · J = {incumbent.value}</p>"
    return _document(
        f"<p><b>Current step:</b> {escape(event.text)}</p>{final}"
        f"<h3>Node {node.id} · {node.status.value}</h3>"
        f"<p><b>Parent:</b> {node.parent if node.parent is not None else '—'}<br>"
        f"<b>Created at step:</b> {node.created_at + 1}<br>"
        f"<b>LP solve order:</b> {node.processed_order if node.processed_order else '—'}</p>"
        f"<p><b>Branching conditions:</b> {escape(branches)}</p>"
        f"<p><b>LP solution:</b> {point_text(node.solution)}<br>"
        f"<b>LP objective:</b> {node.value if node.value is not None else '—'}<br>"
        f"<b>Local bounds:</b> {bounds_text(node.bounds)}<br>"
        f"<b>Global bounds:</b> {bounds_text(frame.event.bounds)}<br>"
        f"<b>Incumbent at this step:</b> {incumbent_info}</p>"
        f"<p><b>Integrality check:</b> {escape(integrality)}<br>"
        f"<b>Branching variable:</b> {escape(branching)}<br>"
        f"<b>Decision:</b> {escape(node.reason)}</p>{_lp(node.problem)}"
    )


def format_relaxation(problem: LinearProgram, result: SimplexResult) -> str:
    value = relaxation_value(problem, result)
    bound = "Upper Bound z<sub>up</sub>" if problem.objective_sense is ObjectiveSense.MAXIMIZE else "Lower Bound z<sub>lower</sub>"
    intro = "<p><b>Automatisch berechnet.</b> Die Ganzzahligkeitsbedingungen sind aufgehoben.</p>"
    if value is None:
        return f"{intro}<p>{escape(result.message)}<br>{bound}: offen</p>"
    point = result.steps[-1].point
    return (
        f"{intro}<p>Optimaler Eckpunkt x<sup>*</sup><sub>LP</sub> = {point_text(point)}<br>"
        f"Zielfunktionswert: {_number(value)}<br><b>{bound} = {_number(value)}</b></p>"
    )


def format_comparison(
    problem: LinearProgram, comparison: PointCheck | None, input_message: str | None
) -> str:
    if comparison is None:
        return f"<p>{escape(input_message or 'Gib alle Koordinaten ein, um den Punkt zu prüfen.')}</p>"
    point = point_text(comparison.point)
    if comparison.value is not None:
        bound = "Lower Bound z<sub>lower</sub>" if problem.objective_sense is ObjectiveSense.MAXIMIZE else "Upper Bound z<sub>upper</sub>"
        return (
            f"<p><b>Zulässiger Punkt:</b> {point}<br>"
            f"Zielfunktionswert (automatisch): {_number(comparison.value)}<br>"
            f"<b>{bound} = {_number(comparison.value)}</b></p>"
        )
    groups = (
        ("Ganzzahligkeit", comparison.integrality_errors),
        ("Variablengrenzen", comparison.domain_errors),
        ("Verletzte Nebenbedingungen", comparison.constraint_errors),
    )
    issues = "".join(
        f"<p><b>{label}</b></p><ul>" + "".join(f"<li>{escape(issue)}</li>" for issue in errors) + "</ul>"
        for label, errors in groups if errors
    )
    return f"<p><b>Unzulässiger Punkt:</b> {point}. Die manuelle Schranke bleibt offen.</p>{issues}"


def format_relaxation_range(
    problem: LinearProgram, result: SimplexResult, comparison: PointCheck | None
) -> str:
    if result.status == "infeasible":
        return "<p>Die LP-Relaxation ist unzulässig. Das ILP besitzt keine zulässige Lösung.</p>"
    bounds = relaxation_bounds(problem, result, comparison)
    lower = _number(bounds.lower) if bounds.lower is not None else "offen"
    upper = _number(bounds.upper) if bounds.upper is not None else "offen"
    optimum = "z<sup>*</sup><sub>ILP</sub>"
    numeric = [str(bounds.lower)] if bounds.lower is not None else []
    numeric.append(optimum)
    if bounds.upper is not None:
        numeric.append(str(bounds.upper))
    inequality = " ≤ ".join(numeric) if len(numeric) > 1 else "Derzeit keine Schranke verfügbar."
    return (
        f"<p>Lower Bound: {lower}<br>Upper Bound: {upper}</p>"
        f"<p>z<sub>lower</sub> ≤ {optimum} ≤ z<sub>upper</sub><br>"
        f"<b>{inequality}</b></p>"
    )


def _number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value)
    try:
        return f"{value} (≈ {float(value):.6g})"
    except OverflowError:
        return str(value)


def format_simplex_trace(result: SimplexResult) -> str:
    sections = [f"<p>{escape(result.message)}</p>"]
    for phase in result.phases:
        names = tuple(row.name for row in phase.rows)
        rows = "\n".join(
            f"{row.name}: {point_text(row.coefficients)} · x ≤ {row.bound}" for row in phase.rows
        )
        sections.append(f"<h3>{escape(phase.name)}</h3><p>Maximization coefficients: "
                        f"{point_text(phase.objective)}</p><pre>{escape(rows)}</pre>")
        for k, calc in enumerate(phase.calculations):
            basis = ", ".join(names[i] for i in calc.step.basis)
            basis_matrix = "\n".join(point_text(phase.rows[i].coefficients) for i in calc.step.basis)
            matrix = "\n".join(point_text(row) for row in calc.inverse)
            gains = ", ".join(f"{names[i]}: {gain}" for i, gain in calc.gains) or "—"
            ratios = ", ".join(f"{names[i]}: {ratio}" for ratio, i in calc.ratios) or "—"
            sections.append(
                f"<p><b>Iteration {k}</b> · x = {point_text(calc.step.point)}<br>"
                f"Basis B: {escape(basis)}<br>Basis matrix E<sub>B</sub></p><pre>{basis_matrix}</pre>"
                "<p>Inverse basis matrix E<sub>B</sub><sup>−1</sup></p>"
                f"<pre>{matrix}</pre><p>Tested gains cᵀd: {escape(gains)}<br>"
                f"Direction d: {point_text(calc.direction)}<br>"
                f"Blocking ratios: {escape(ratios)}<br><b>{escape(calc.decision)}</b></p>"
            )
    return _document("".join(sections))
