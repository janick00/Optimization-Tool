"""Linear-programming problem parsing, validation, and form conversion.

The four representations implemented here follow Chapter 3.1 of the course
slides: General Form, Canonical Form, Standard Form, and Inequality Form.
The module deliberately has no Qt dependencies so it can be used independently
of the graphical user interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from html import escape as html_escape
from typing import Sequence


class ObjectiveSense(str, Enum):
    """Direction of the linear objective function."""

    MAXIMIZE = "max"
    MINIMIZE = "min"

    @property
    def label(self) -> str:
        return "Maximize" if self is ObjectiveSense.MAXIMIZE else "Minimize"


class ConstraintRelation(str, Enum):
    """Relation used by one linear constraint."""

    LESS_EQUAL = "<="
    EQUAL = "="
    GREATER_EQUAL = ">="


class VariableDomain(str, Enum):
    """Allowed sign/domain of one decision variable."""

    NONNEGATIVE = ">=0"
    FREE = "free"
    NONPOSITIVE = "<=0"


class ProblemForm(str, Enum):
    """LP forms from the problem-formulation lecture slides."""

    GENERAL = "general"
    CANONICAL = "canonical"
    STANDARD = "standard"
    INEQUALITY = "inequality"

    @property
    def label(self) -> str:
        return {
            ProblemForm.GENERAL: "General Form",
            ProblemForm.CANONICAL: "Canonical Form",
            ProblemForm.STANDARD: "Standard Form",
            ProblemForm.INEQUALITY: "Inequality Form",
        }[self]


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


@dataclass(frozen=True)
class LinearConstraint:
    """One row of a linear constraint system."""

    coefficients: tuple[Fraction, ...]
    relation: ConstraintRelation
    right_hand_side: Fraction
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            tuple(_as_fraction(value) for value in self.coefficients),
        )
        object.__setattr__(self, "relation", ConstraintRelation(self.relation))
        object.__setattr__(self, "right_hand_side", _as_fraction(self.right_hand_side))


@dataclass(frozen=True)
class LinearProgram:
    """Validated internal representation of a linear program."""

    objective_sense: ObjectiveSense
    objective: tuple[Fraction, ...]
    constraints: tuple[LinearConstraint, ...]
    variable_names: tuple[str, ...]
    variable_domains: tuple[VariableDomain, ...]

    def __post_init__(self) -> None:
        objective = tuple(_as_fraction(value) for value in self.objective)
        names = tuple(str(name).strip() for name in self.variable_names)
        domains = tuple(VariableDomain(domain) for domain in self.variable_domains)
        constraints = tuple(
            constraint
            if isinstance(constraint, LinearConstraint)
            else LinearConstraint(**constraint)
            for constraint in self.constraints
        )

        if not objective:
            raise ValueError("At least one objective coefficient is required.")
        if len(names) != len(objective):
            raise ValueError("The number of variable names must match the objective.")
        if len(domains) != len(objective):
            raise ValueError("The number of variable domains must match the objective.")
        if len(set(names)) != len(names):
            raise ValueError("Variable names must be unique.")
        if any(not name for name in names):
            raise ValueError("Variable names cannot be empty.")
        if any(len(constraint.coefficients) != len(objective) for constraint in constraints):
            raise ValueError("Every constraint row must match the number of variables.")

        object.__setattr__(self, "objective_sense", ObjectiveSense(self.objective_sense))
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "variable_names", names)
        object.__setattr__(self, "variable_domains", domains)


@dataclass(frozen=True)
class ConversionResult:
    """Converted LP together with human-readable transformation steps."""

    problem: LinearProgram
    source_form: ProblemForm
    target_form: ProblemForm
    steps: tuple[str, ...]


def _as_fraction(value: Fraction | int | float | str) -> Fraction:
    """Convert common numeric input into an exact rational value."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Empty values are not valid numbers.")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid number: {value}") from error


def _split_numeric_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[,;\s]+", text.strip()) if token]


def parse_vector(text: str, field_name: str) -> tuple[Fraction, ...]:
    """Parse a comma- or whitespace-separated numeric vector."""

    tokens = _split_numeric_tokens(text)
    if not tokens:
        raise ValueError(f"{field_name} must contain at least one value.")
    try:
        return tuple(_as_fraction(token) for token in tokens)
    except ValueError as error:
        raise ValueError(f"Invalid value in {field_name}: {error}") from error


def parse_matrix(text: str, field_name: str = "Constraint matrix A") -> tuple[tuple[Fraction, ...], ...]:
    """Parse one numeric coefficient row per non-empty input line."""

    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{field_name} must contain at least one row.")

    matrix = tuple(parse_vector(row, f"{field_name}, row {index}") for index, row in enumerate(rows, 1))
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"Every row in {field_name} must have the same number of values.")
    return matrix


def parse_relations(text: str, expected_count: int) -> tuple[ConstraintRelation, ...]:
    """Parse relation symbols such as ``<=, =, >=``."""

    symbols = re.findall(r"<=|>=|=|≤|≥", text)
    if len(symbols) != expected_count:
        raise ValueError(
            f"Constraint relations must contain exactly {expected_count} symbols."
        )
    normalised = {"≤": "<=", "≥": ">="}
    return tuple(ConstraintRelation(normalised.get(symbol, symbol)) for symbol in symbols)


def parse_variable_domains(text: str, expected_count: int) -> tuple[VariableDomain, ...]:
    """Parse variable-domain tokens such as ``>=0, free, <=0``."""

    tokens = [
        token.strip().lower().replace(" ", "")
        for token in re.split(r"[,;]+", text)
        if token.strip()
    ]
    if len(tokens) == 1 and expected_count > 1:
        tokens = [token for token in re.split(r"\s+", tokens[0]) if token]
    if len(tokens) != expected_count:
        raise ValueError(f"Variable domains must contain exactly {expected_count} values.")

    aliases = {
        ">=0": VariableDomain.NONNEGATIVE,
        "nonnegative": VariableDomain.NONNEGATIVE,
        "positive": VariableDomain.NONNEGATIVE,
        "free": VariableDomain.FREE,
        "r": VariableDomain.FREE,
        "real": VariableDomain.FREE,
        "<=0": VariableDomain.NONPOSITIVE,
        "nonpositive": VariableDomain.NONPOSITIVE,
        "negative": VariableDomain.NONPOSITIVE,
    }
    try:
        return tuple(aliases[token] for token in tokens)
    except KeyError as error:
        raise ValueError("Variable domains must use >=0, free, or <=0.") from error


def parse_variable_names(text: str, variable_count: int) -> tuple[str, ...]:
    """Parse optional variable names, or generate x1, x2, ... automatically."""

    if not text.strip():
        return tuple(f"x{index}" for index in range(1, variable_count + 1))
    names = tuple(token for token in re.split(r"[,;\s]+", text.strip()) if token)
    if len(names) != variable_count:
        raise ValueError(f"Variable names must contain exactly {variable_count} values.")
    return names


def build_linear_program(
    source_form: ProblemForm | str,
    objective_sense: ObjectiveSense | str,
    objective_text: str,
    matrix_text: str,
    right_hand_side_text: str,
    relation_text: str = "",
    domain_text: str = "",
    variable_names_text: str = "",
) -> LinearProgram:
    """Build an internal LP from the fields used by the GUI editor."""

    source_form = ProblemForm(source_form)
    objective_sense = ObjectiveSense(objective_sense)
    objective = parse_vector(objective_text, "Objective coefficients c")
    matrix = parse_matrix(matrix_text)
    right_hand_side = parse_vector(right_hand_side_text, "Right-hand side b")

    if len(matrix[0]) != len(objective):
        raise ValueError("Every constraint row must match the objective length.")
    if len(right_hand_side) != len(matrix):
        raise ValueError("The right-hand side must contain one value per constraint row.")

    variable_names = parse_variable_names(variable_names_text, len(objective))
    if source_form is ProblemForm.GENERAL:
        relations = parse_relations(relation_text, len(matrix))
        domains = parse_variable_domains(domain_text, len(objective))
    elif source_form is ProblemForm.STANDARD:
        relations = (ConstraintRelation.EQUAL,) * len(matrix)
        domains = (VariableDomain.NONNEGATIVE,) * len(objective)
    elif source_form is ProblemForm.CANONICAL:
        relation = _objective_relation(objective_sense)
        relations = (relation,) * len(matrix)
        domains = (VariableDomain.NONNEGATIVE,) * len(objective)
    else:
        relation = _objective_relation(objective_sense)
        relations = (relation,) * len(matrix)
        domains = (VariableDomain.FREE,) * len(objective)

    constraints = tuple(
        LinearConstraint(tuple(row), relation, rhs, label=f"Constraint {index}")
        for index, (row, relation, rhs) in enumerate(
            zip(matrix, relations, right_hand_side), 1
        )
    )
    return LinearProgram(
        objective_sense=objective_sense,
        objective=objective,
        constraints=constraints,
        variable_names=variable_names,
        variable_domains=domains,
    )


def convert_problem(
    problem: LinearProgram,
    target_form: ProblemForm | str,
    source_form: ProblemForm | str = ProblemForm.GENERAL,
) -> ConversionResult:
    """Convert a problem into one of the four lecture-slide forms."""

    target_form = ProblemForm(target_form)
    source_form = ProblemForm(source_form)

    if target_form is ProblemForm.GENERAL:
        converted = problem
    elif target_form is ProblemForm.STANDARD:
        converted = _to_standard_form(problem)
    elif target_form is ProblemForm.CANONICAL:
        converted = _to_canonical_form(problem)
    else:
        converted = _to_inequality_form(problem)

    return ConversionResult(
        problem=converted,
        source_form=source_form,
        target_form=target_form,
        steps=_build_conversion_steps(problem, source_form, target_form),
    )


def _objective_relation(sense: ObjectiveSense) -> ConstraintRelation:
    return (
        ConstraintRelation.LESS_EQUAL
        if sense is ObjectiveSense.MAXIMIZE
        else ConstraintRelation.GREATER_EQUAL
    )


def _opposite_relation(relation: ConstraintRelation) -> ConstraintRelation:
    if relation is ConstraintRelation.LESS_EQUAL:
        return ConstraintRelation.GREATER_EQUAL
    if relation is ConstraintRelation.GREATER_EQUAL:
        return ConstraintRelation.LESS_EQUAL
    return ConstraintRelation.EQUAL


def _negate_constraint(constraint: LinearConstraint) -> LinearConstraint:
    return LinearConstraint(
        tuple(-coefficient for coefficient in constraint.coefficients),
        _opposite_relation(constraint.relation),
        -constraint.right_hand_side,
        constraint.label,
    )


def _to_required_direction(
    constraint: LinearConstraint,
    required_relation: ConstraintRelation,
) -> tuple[LinearConstraint, ...]:
    """Convert one constraint into <= or >=, splitting equalities if needed."""

    if constraint.relation is required_relation:
        return (constraint,)
    if constraint.relation is _opposite_relation(required_relation):
        return (_negate_constraint(constraint),)
    return (
        LinearConstraint(
            constraint.coefficients,
            required_relation,
            constraint.right_hand_side,
            constraint.label,
        ),
        LinearConstraint(
            tuple(-coefficient for coefficient in constraint.coefficients),
            required_relation,
            -constraint.right_hand_side,
            constraint.label,
        ),
    )


def _substitute_variables(
    problem: LinearProgram,
) -> tuple[tuple[Fraction, ...], tuple[str, ...], tuple[LinearConstraint, ...]]:
    """Replace signed/free variables by non-negative variables."""

    references: list[tuple[tuple[int, Fraction], ...]] = []
    names: list[str] = []
    for name, domain in zip(problem.variable_names, problem.variable_domains):
        if domain is VariableDomain.NONNEGATIVE:
            references.append(((len(names), Fraction(1)),))
            names.append(name)
        elif domain is VariableDomain.NONPOSITIVE:
            references.append(((len(names), Fraction(-1)),))
            names.append(f"{name}_neg")
        else:
            positive_index = len(names)
            names.extend((f"{name}_plus", f"{name}_minus"))
            references.append(
                (
                    (positive_index, Fraction(1)),
                    (positive_index + 1, Fraction(-1)),
                )
            )

    objective = [Fraction(0)] * len(names)
    for coefficient, variable_references in zip(problem.objective, references):
        for index, multiplier in variable_references:
            objective[index] += coefficient * multiplier

    constraints: list[LinearConstraint] = []
    for constraint in problem.constraints:
        row = [Fraction(0)] * len(names)
        for coefficient, variable_references in zip(constraint.coefficients, references):
            for index, multiplier in variable_references:
                row[index] += coefficient * multiplier
        constraints.append(
            LinearConstraint(tuple(row), constraint.relation, constraint.right_hand_side, constraint.label)
        )
    return tuple(objective), tuple(names), tuple(constraints)


def _to_standard_form(problem: LinearProgram) -> LinearProgram:
    objective, names, base_constraints = _substitute_variables(problem)
    output_names = list(names)
    output_objective = list(objective)
    output_constraints: list[list[Fraction]] = []
    extra_variable_count = 0

    for index, constraint in enumerate(base_constraints, 1):
        row = list(constraint.coefficients) + [Fraction(0)] * extra_variable_count
        if constraint.relation is not ConstraintRelation.EQUAL:
            for previous_row in output_constraints:
                previous_row.append(Fraction(0))
            variable_name = (
                f"s_{index}"
                if constraint.relation is ConstraintRelation.LESS_EQUAL
                else f"u_{index}"
            )
            output_names.append(variable_name)
            output_objective.append(Fraction(0))
            row.append(
                Fraction(1)
                if constraint.relation is ConstraintRelation.LESS_EQUAL
                else Fraction(-1)
            )
            extra_variable_count += 1
        output_constraints.append(row)

    constraints = tuple(
        LinearConstraint(tuple(row), ConstraintRelation.EQUAL, source.right_hand_side, source.label)
        for row, source in zip(output_constraints, base_constraints)
    )

    return LinearProgram(
        problem.objective_sense,
        tuple(output_objective),
        constraints,
        tuple(output_names),
        (VariableDomain.NONNEGATIVE,) * len(output_names),
    )


def _to_canonical_form(problem: LinearProgram) -> LinearProgram:
    objective, names, base_constraints = _substitute_variables(problem)
    required_relation = _objective_relation(problem.objective_sense)
    constraints = tuple(
        converted
        for constraint in base_constraints
        for converted in _to_required_direction(constraint, required_relation)
    )
    return LinearProgram(
        problem.objective_sense,
        objective,
        constraints,
        names,
        (VariableDomain.NONNEGATIVE,) * len(names),
    )


def _to_inequality_form(problem: LinearProgram) -> LinearProgram:
    required_relation = _objective_relation(problem.objective_sense)
    constraints = list(problem.constraints)
    variable_count = len(problem.variable_names)

    for index, domain in enumerate(problem.variable_domains):
        if domain is VariableDomain.FREE:
            continue
        row = [Fraction(0)] * variable_count
        row[index] = Fraction(1)
        constraints.append(
            LinearConstraint(
                tuple(row),
                ConstraintRelation.GREATER_EQUAL
                if domain is VariableDomain.NONNEGATIVE
                else ConstraintRelation.LESS_EQUAL,
                Fraction(0),
                f"Domain of {problem.variable_names[index]}",
            )
        )

    converted_constraints = tuple(
        converted
        for constraint in constraints
        for converted in _to_required_direction(constraint, required_relation)
    )
    return LinearProgram(
        problem.objective_sense,
        problem.objective,
        converted_constraints,
        problem.variable_names,
        (VariableDomain.FREE,) * variable_count,
    )


def _build_conversion_steps(
    problem: LinearProgram,
    source_form: ProblemForm,
    target_form: ProblemForm,
) -> tuple[str, ...]:
    if source_form is target_form:
        return (f"The problem is already in {target_form.label}.",)

    steps: list[str] = [f"Read the input as {source_form.label}."]
    if target_form in (ProblemForm.CANONICAL, ProblemForm.STANDARD):
        for name, domain in zip(problem.variable_names, problem.variable_domains):
            if domain is VariableDomain.NONPOSITIVE:
                steps.append(f"Substitute {name} = -{name}_neg so {name}_neg >= 0.")
            elif domain is VariableDomain.FREE:
                steps.append(
                    f"Substitute {name} = {name}_plus - {name}_minus with both new variables >= 0."
                )

    if target_form is ProblemForm.STANDARD:
        for index, constraint in enumerate(problem.constraints, 1):
            if constraint.relation is ConstraintRelation.LESS_EQUAL:
                steps.append(f"Constraint {index}: add slack variable s_{index} >= 0.")
            elif constraint.relation is ConstraintRelation.GREATER_EQUAL:
                steps.append(f"Constraint {index}: subtract surplus variable u_{index} >= 0.")
        steps.append("All constraints are now equalities and all variables are non-negative.")
    elif target_form is ProblemForm.CANONICAL:
        required = _objective_relation(problem.objective_sense).value
        steps.append(f"Convert all constraints to the {required} direction.")
        if any(constraint.relation is ConstraintRelation.EQUAL for constraint in problem.constraints):
            steps.append("Replace each equality by two inequalities in the required direction.")
        steps.append("All variables are now non-negative.")
    elif target_form is ProblemForm.INEQUALITY:
        steps.append("Move non-negative/non-positive variable restrictions into the constraint system.")
        required = _objective_relation(problem.objective_sense).value
        steps.append(f"Convert all constraints to the {required} direction; equalities become two inequalities.")
        steps.append("The resulting variables are unrestricted in sign.")
    else:
        steps.append("Keep all constraints and variable-domain declarations in the general form.")

    return tuple(steps)


def format_linear_program(
    problem: LinearProgram,
    form: ProblemForm | str,
    notation: DisplayNotation | str = DisplayNotation.EXPANDED,
) -> str:
    """Render an LP in readable text for the result pane."""

    form = ProblemForm(form)
    notation = DisplayNotation(notation)
    if notation is DisplayNotation.ROW:
        return _format_row_notation(problem, form)
    if notation is DisplayNotation.COLUMN:
        return _format_column_notation(problem, form)
    if notation is DisplayNotation.MATRIX:
        return _format_matrix_notation(problem, form)

    lines = [
        form.label,
        "",
        f"{problem.objective_sense.label}: "
        f"{_format_expression(problem.objective, problem.variable_names)}",
        "Subject to:",
    ]
    if problem.constraints:
        lines.extend(
            f"  {_format_expression(constraint.coefficients, problem.variable_names)} "
            f"{constraint.relation.value} {_format_fraction(constraint.right_hand_side)}"
            for constraint in problem.constraints
        )
    else:
        lines.append("  (no constraints)")
    lines.append("")
    lines.append("Variable domains:")
    lines.extend(
        f"  {name} {domain.value}"
        for name, domain in zip(problem.variable_names, problem.variable_domains)
    )
    return "\n".join(lines)


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


def _format_row_notation(problem: LinearProgram, form: ProblemForm) -> str:
    lines = [
        f"{form.label} - Row notation",
        "",
        f"{problem.objective_sense.label}: c^T x",
        f"  c^T = {_format_vector(problem.objective)}",
        "Subject to:",
    ]
    lines.extend(
        f"  a_{index}^T x {constraint.relation.value} b_{index}"
        f"    with a_{index}^T = {_format_vector(constraint.coefficients)}"
        f", b_{index} = {_format_fraction(constraint.right_hand_side)}"
        for index, constraint in enumerate(problem.constraints, 1)
    )
    lines.extend(("", "Variable domains:"))
    lines.append("  x_j follows the individual domains below, j = 1, ..., n")
    lines.extend(
        f"  {name} {domain.value}"
        for name, domain in zip(problem.variable_names, problem.variable_domains)
    )
    return "\n".join(lines)


def _format_column_notation(problem: LinearProgram, form: ProblemForm) -> str:
    lines = [
        f"{form.label} - Column notation",
        "",
        f"{problem.objective_sense.label}: sum(j=1..n) c_j x_j",
        f"  = {_format_expression(problem.objective, problem.variable_names)}",
        "Subject to:",
    ]
    for index, constraint in enumerate(problem.constraints, 1):
        lines.append(
            f"  sum(j=1..n) a_{index}j x_j {constraint.relation.value} b_{index}"
            f"    with b_{index} = {_format_fraction(constraint.right_hand_side)}"
        )
    lines.extend(("", "Columns of A:"))
    for index, column in enumerate(_matrix_columns(problem), 1):
        lines.append(f"  A_{index} = {_format_vector(column, transpose=True)}")
    lines.extend(("", "Variable domains:"))
    lines.extend(
        f"  {name} {domain.value}"
        for name, domain in zip(problem.variable_names, problem.variable_domains)
    )
    return "\n".join(lines)


def _format_matrix_notation(problem: LinearProgram, form: ProblemForm) -> str:
    lines = [
        f"{form.label} - Matrix notation",
        "",
        f"{problem.objective_sense.label}: c^T x",
        f"  c = {_format_vector(problem.objective, transpose=True)}",
        f"  x = {_format_vector(problem.variable_names, transpose=True)}",
        f"  b = {_format_vector([constraint.right_hand_side for constraint in problem.constraints], transpose=True)}",
        "  A =",
    ]
    lines.extend(f"    {_format_vector(constraint.coefficients)}" for constraint in problem.constraints)
    lines.append("Subject to:")
    relation_values = {constraint.relation.value for constraint in problem.constraints}
    if len(relation_values) == 1:
        relation = next(iter(relation_values))
        lines.append(f"  A x {relation} b")
    else:
        lines.append("  Row-wise relations:")
        lines.extend(
            f"    (A x)_{index} {constraint.relation.value} b_{index}"
            for index, constraint in enumerate(problem.constraints, 1)
        )
    lines.extend(("", "Variable domains:"))
    lines.extend(
        f"  {name} {domain.value}"
        for name, domain in zip(problem.variable_names, problem.variable_domains)
    )
    return "\n".join(lines)


def _matrix_columns(problem: LinearProgram) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(constraint.coefficients[index] for constraint in problem.constraints)
        for index in range(len(problem.variable_names))
    )


def _format_vector(values: Sequence[Fraction | str], transpose: bool = False) -> str:
    formatted_values = ", ".join(
        _format_fraction(value) if not isinstance(value, str) else value for value in values
    )
    suffix = "^T" if transpose else ""
    return f"({formatted_values}){suffix}"


def format_conversion_steps(result: ConversionResult) -> str:
    """Render the conversion trace for the result pane."""

    return "\n".join(f"{index}. {step}" for index, step in enumerate(result.steps, 1))


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _format_expression(coefficients: Sequence[Fraction], variable_names: Sequence[str]) -> str:
    terms: list[str] = []
    for coefficient, name in zip(coefficients, variable_names):
        coefficient = _as_fraction(coefficient)
        if coefficient == 0:
            continue
        absolute = abs(coefficient)
        coefficient_text = "" if absolute == 1 else f"{_format_fraction(absolute)} "
        term = f"{coefficient_text}{name}"
        if not terms:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            terms.append(f"- {term}" if coefficient < 0 else f"+ {term}")
    return " ".join(terms) if terms else "0"
