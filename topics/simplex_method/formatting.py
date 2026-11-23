"""Render the exact start checks and simplex pivot history."""

from html import escape

from topics.linear_programming.models import LinearProgram
from topics.simplex_method.solver import SimplexResult, objective_value


def format_calculations(problem: LinearProgram, result: SimplexResult) -> str:
    start = result.start
    sections = [f"<p><b>Mode:</b> {escape(result.mode)}</p>"]
    if start is None:
        sections.append(f"<p>{escape(result.message)}</p>")
    else:
        active = ", ".join(check.row.name for check in start.checks if check.active)
        violations = any(not check.satisfied for check in start.checks)
        sections.extend((
            f"<p><b>Starting point:</b> {_point(start.point)}</p>",
            "<h3>Feasibility check</h3>",
            "<p>All rows are checked as E<sub>i</sub>x ≤ b<sub>i</sub>, including variable domains.</p>",
            f"<p>{'Infeasible' if violations else 'Feasible'}.</p>",
            "<table><tr><th>Row</th><th>E<sub>i</sub>x</th><th>b<sub>i</sub></th><th>Check</th></tr>",
        ))
        for check in start.checks:
            row_type = " (domain)" if check.row.is_domain else ""
            status = "Active" if check.active else "Satisfied" if check.satisfied else "Violated"
            sections.append(
                f"<tr><td>{escape(check.row.name)}{row_type}</td><td>{check.value}</td>"
                f"<td>{check.row.bound}</td><td>{status}</td></tr>"
            )
        sections.extend((
            "</table>",
            f"<p><b>Active constraints:</b> {escape(active) if active else 'None'}</p>",
            f"<p><b>Linear independence:</b> active rank = {start.active_rank}, "
            f"required rank = {len(start.point)}.</p>",
            f"<p><b>Start objective value:</b> {start.objective_value}</p>",
        ))
        if start.error:
            sections.append(f"<p><b>{escape(start.error)}</b></p>")
        else:
            names = tuple(check.row.name for check in start.checks)
            basis = ", ".join(names[i] for i in start.basis)
            matrix = "\n".join("[ " + "  ".join(map(str, row)) + " ]" for row in start.matrix)
            sections.extend((
                f"<p><b>Start basis B:</b> {escape(basis)}</p>",
                f"<p><b>Basis matrix E<sub>B</sub>:</b></p><pre>{matrix}</pre>",
                f"<p>rank(E<sub>B</sub>) = {len(start.basis)}. Selected the first independent "
                "active rows in input order, with variable domains after the constraints.</p>",
                "<h3>Simplex pivots</h3>",
                "<table><tr><th>k</th><th>Point</th><th>Objective</th><th>Pivot</th></tr>",
            ))
            for i, step in enumerate(result.steps):
                change = (
                    f"Release {escape(names[step.released])}, add {escape(names[step.added])}; "
                    f"step {step.distance}"
                    if step.released is not None and step.added is not None else "Starting basis"
                )
                sections.append(
                    f"<tr><td>{i}</td><td>{_point(step.point)}</td>"
                    f"<td>{objective_value(problem, step.point)}</td><td>{change}</td></tr>"
                )
            sections.extend(("</table>", f"<p>{escape(result.message)}</p>"))
    return (
        "<html><head><style>"
        "body { color: #dbeafe; font-size: 10pt; }"
        "th, h3, b { color: #93c5fd; }"
        "td, th { padding: 3px 6px; border-bottom: 1px solid #334155; }"
        "</style></head><body>" + "".join(sections) + "</body></html>"
    )


def _point(point) -> str:
    return "(" + ", ".join(map(str, point)) + ")"
