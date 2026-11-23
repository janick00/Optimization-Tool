"""HTML representations of linear programs."""

from __future__ import annotations

import re
from enum import Enum
from fractions import Fraction
from html import escape as html_escape
from typing import Sequence

from topics.linear_programming.models import (
    ConversionResult,
    LinearProgram,
    ProblemForm,
    _as_fraction,
)


class DisplayNotation(str, Enum):
    """Notation used when rendering a converted LP for the user."""

    EXPANDED = "expanded"
    ROW = "row"
    COLUMN = "column"
    MATRIX = "matrix"

    @property
    def label(self) -> str:
        return {
            DisplayNotation.EXPANDED: "Expanded notation",
            DisplayNotation.ROW: "Row notation",
            DisplayNotation.COLUMN: "Column notation",
            DisplayNotation.MATRIX: "Matrix notation",
        }[self]


def format_linear_program_html(
    problem: LinearProgram,
    form: ProblemForm | str,
    notation: DisplayNotation | str = DisplayNotation.EXPANDED,
) -> str:
    """Render an LP as styled rich text with LaTeX-like typography."""

    form = ProblemForm(form)
    notation = DisplayNotation(notation)
    title = f"{html_escape(form.label)}"
    if notation is DisplayNotation.ROW:
        title += " <span class='notation'>· Row notation</span>"
        body = _html_row_notation_body(problem)
    elif notation is DisplayNotation.COLUMN:
        title += " <span class='notation'>· Column notation</span>"
        body = _html_column_notation_body(problem)
    elif notation is DisplayNotation.MATRIX:
        title += " <span class='notation'>· Matrix notation</span>"
        body = _html_matrix_notation_body(problem)
    else:
        body = _html_expanded_body(problem)

    return _html_document(
        f"<div class='form-title'>{title}</div>{body}"
    )


def format_conversion_steps_html(result: ConversionResult) -> str:
    """Render conversion steps as a styled rich-text list."""

    items = "".join(f"<li>{html_escape(step)}</li>" for step in result.steps)
    return _html_document(
        "<div class='steps-title'>Conversion steps</div>"
        f"<ol class='steps'>{items}</ol>"
    )


def _html_document(body: str) -> str:
    return f"""<html>
<head>
<style>
body {{
    color: #dbeafe;
    background-color: #0f172a;
    font-family: 'DejaVu Serif', 'Liberation Serif', serif;
    font-size: 11pt;
}}
.form-title {{
    color: #f8fafc;
    font-size: 14pt;
    font-weight: bold;
    margin-bottom: 12px;
}}
.notation {{
    color: #93c5fd;
    font-size: 10pt;
    font-weight: normal;
}}
.section-title, .steps-title {{
    color: #93c5fd;
    font-size: 10pt;
    font-weight: bold;
    margin-top: 12px;
    margin-bottom: 5px;
}}
.equation {{
    color: #f8fafc;
    font-size: 12pt;
    padding: 3px 0;
}}
.operator {{
    color: #c4b5fd;
    font-weight: bold;
}}
.detail {{
    color: #94a3b8;
    font-size: 9pt;
}}
.constraints td {{
    padding: 3px 4px;
    vertical-align: top;
}}
.constraint-index {{
    color: #64748b;
    font-size: 9pt;
}}
.domain {{
    color: #e0e7ff;
    padding: 2px 0;
}}
.matrix td {{
    color: #f8fafc;
    padding: 2px 8px;
    text-align: center;
}}
.matrix-bracket {{
    color: #93c5fd;
    font-size: 24pt;
    padding: 0 2px;
}}
.matrix-wrapper td {{
    vertical-align: middle;
}}
.definitions {{
    color: #94a3b8;
    font-size: 9pt;
    padding: 2px 0;
}}
.steps {{
    color: #cbd5e1;
    margin-top: 4px;
}}
.steps li {{
    padding: 2px 0;
}}
</style>
</head>
<body>{body}</body>
</html>"""


def _html_expanded_body(problem: LinearProgram) -> str:
    constraint_rows = "".join(
        "<tr>"
        f"<td class='constraint-index'>({index})</td>"
        f"<td class='equation'>{_html_expression(constraint.coefficients, problem.variable_names)} "
        f"{_html_relation(constraint.relation.value)} "
        f"{_html_number(constraint.right_hand_side)}</td>"
        "</tr>"
        for index, constraint in enumerate(problem.constraints, 1)
    )
    return (
        f"<div class='equation'><span class='operator'>{problem.objective_sense.label}</span>&nbsp; "
        f"{_html_expression(problem.objective, problem.variable_names)}</div>"
        "<div class='section-title'>Subject to</div>"
        f"<table class='constraints' cellspacing='0'>{constraint_rows}</table>"
        "<div class='section-title'>Variable domains</div>"
        f"{_html_domains(problem)}"
    )


def _html_row_notation_body(problem: LinearProgram) -> str:
    rows = "".join(
        "<tr>"
        f"<td class='constraint-index'>({index})</td>"
        f"<td class='equation'><i>a</i><sub>{index}</sub><sup>T</sup><i>x</i> "
        f"{_html_relation(constraint.relation.value)} "
        f"<i>b</i><sub>{index}</sub> "
        f"<span class='detail'>with <i>a</i><sub>{index}</sub><sup>T</sup> = "
        f"{_html_vector(constraint.coefficients)}, "
        f"<i>b</i><sub>{index}</sub> = {_html_number(constraint.right_hand_side)}</span></td>"
        "</tr>"
        for index, constraint in enumerate(problem.constraints, 1)
    )
    return (
        f"<div class='equation'><span class='operator'>{problem.objective_sense.label}</span>&nbsp; "
        "<i>c</i><sup>T</sup><i>x</i> "
        f"<span class='detail'>with <i>c</i><sup>T</sup> = {_html_vector(problem.objective)}</span></div>"
        "<div class='section-title'>Subject to</div>"
        f"<table class='constraints' cellspacing='0'>{rows}</table>"
        "<div class='section-title'>Variable domains</div>"
        f"{_html_domains(problem)}"
    )


def _html_column_notation_body(problem: LinearProgram) -> str:
    rows = "".join(
        "<tr>"
        f"<td class='constraint-index'>({index})</td>"
        f"<td class='equation'>&sum;<sub>j=1</sub><sup>n</sup> "
        f"a<sub>{index}j</sub>x<sub>j</sub> { _html_relation(constraint.relation.value) } "
        f"b<sub>{index}</sub> = {_html_number(constraint.right_hand_side)}</td>"
        "</tr>"
        for index, constraint in enumerate(problem.constraints, 1)
    )
    columns = "".join(
        f"<span class='detail'><i>A</i><sub>{index}</sub> = "
        f"{_html_vector(column, transpose=True)}</span><br>"
        for index, column in enumerate(_matrix_columns(problem), 1)
    )
    return (
        f"<div class='equation'><span class='operator'>{problem.objective_sense.label}</span>&nbsp; "
        "&sum;<sub>j=1</sub><sup>n</sup> <i>c</i><sub>j</sub>x<sub>j</sub> "
        f"<span class='detail'>= {_html_expression(problem.objective, problem.variable_names)}</span></div>"
        "<div class='section-title'>Subject to</div>"
        f"<table class='constraints' cellspacing='0'>{rows}</table>"
        "<div class='section-title'>Columns of A</div>"
        f"<div class='definitions'>{columns}</div>"
        "<div class='section-title'>Variable domains</div>"
        f"{_html_domains(problem)}"
    )


def _html_matrix_notation_body(problem: LinearProgram) -> str:
    relation_values = {constraint.relation.value for constraint in problem.constraints}
    if len(relation_values) == 1:
        relation_body = (
            f"<div class='equation'><i>A</i><i>x</i> "
            f"{_html_relation(next(iter(relation_values)))} <i>b</i></div>"
        )
    else:
        rows = "".join(
            f"<div class='equation'>(<i>A</i><i>x</i>)<sub>{index}</sub> "
            f"{_html_relation(constraint.relation.value)} "
            f"<i>b</i><sub>{index}</sub></div>"
            for index, constraint in enumerate(problem.constraints, 1)
        )
        relation_body = f"<div class='detail'>Row-wise relations</div>{rows}"
    return (
        f"<div class='equation'><span class='operator'>{problem.objective_sense.label}</span>&nbsp; "
        "<i>c</i><sup>T</sup><i>x</i></div>"
        f"<div class='definitions'><i>c</i> = {_html_vector(problem.objective, transpose=True)}<br>"
        f"<i>x</i> = {_html_vector(problem.variable_names, transpose=True)}<br>"
        f"<i>b</i> = {_html_vector([constraint.right_hand_side for constraint in problem.constraints], transpose=True)}</div>"
        "<div class='section-title'>Matrix A</div>"
        f"{_html_matrix([constraint.coefficients for constraint in problem.constraints])}"
        "<div class='section-title'>Subject to</div>"
        f"{relation_body}"
        "<div class='section-title'>Variable domains</div>"
        f"{_html_domains(problem)}"
    )


def _html_domains(problem: LinearProgram) -> str:
    domains = "".join(
        f"<div class='domain'>{_html_variable_name(name)} "
        f"{_html_relation(domain.value)}</div>"
        for name, domain in zip(problem.variable_names, problem.variable_domains)
    )
    return domains


def _html_matrix(rows: Sequence[Sequence[Fraction]]) -> str:
    row_html = "".join(
        "<tr>" + "".join(f"<td>{_html_number(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        "<table class='matrix-wrapper' cellspacing='0'><tr>"
        "<td class='matrix-bracket'>[</td>"
        f"<td><table class='matrix' cellspacing='0'>{row_html}</table></td>"
        "<td class='matrix-bracket'>]</td>"
        "</tr></table>"
    )


def _html_vector(values: Sequence[Fraction | str], transpose: bool = False) -> str:
    formatted = ", ".join(
        _html_number(value) if not isinstance(value, str) else _html_variable_name(value)
        for value in values
    )
    suffix = "<sup>T</sup>" if transpose else ""
    return f"({formatted}){suffix}"


def _html_expression(coefficients: Sequence[Fraction], variable_names: Sequence[str]) -> str:
    terms: list[str] = []
    for coefficient, name in zip(coefficients, variable_names):
        coefficient = _as_fraction(coefficient)
        if coefficient == 0:
            continue
        absolute = abs(coefficient)
        coefficient_text = "" if absolute == 1 else f"{_html_number(absolute)}&nbsp;"
        term = f"{coefficient_text}{_html_variable_name(name)}"
        if not terms:
            terms.append(f"&minus;&nbsp;{term}" if coefficient < 0 else term)
        else:
            terms.append(f"&nbsp;&minus;&nbsp;{term}" if coefficient < 0 else f"&nbsp;+&nbsp;{term}")
    return "".join(terms) if terms else "0"


def _html_variable_name(name: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)(?:_(\d+)|(\d+))(?:_(plus|minus|neg))?", name)
    if not match:
        return f"<i>{html_escape(name)}</i>"
    base, underscored_index, plain_index, suffix = match.groups()
    index = underscored_index or plain_index
    suffix_text = f", {suffix}" if suffix else ""
    return f"<i>{html_escape(base)}</i><sub>{index}{html_escape(suffix_text)}</sub>"


def _html_number(value: Fraction | int | float | str) -> str:
    return html_escape(_format_fraction(_as_fraction(value)))


def _html_relation(relation: str) -> str:
    return {
        "<=": "&le;",
        ">=": "&ge;",
        "<=0": "&le;&nbsp;0",
        ">=0": "&ge;&nbsp;0",
        "=": "=",
        "free": "&isin;&nbsp;ℝ",
    }[relation]


def _matrix_columns(problem: LinearProgram) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(constraint.coefficients[index] for constraint in problem.constraints)
        for index in range(len(problem.variable_names))
    )


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
