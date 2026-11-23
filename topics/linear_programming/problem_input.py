"""Read and validate the text fields that describe a linear program."""

from __future__ import annotations

import re
from fractions import Fraction

from topics.linear_programming.form_conversion import form_defaults
from topics.linear_programming.models import (
    ConstraintRelation,
    LinearConstraint,
    LinearProgram,
    ObjectiveSense,
    ProblemForm,
    VariableDomain,
    _as_fraction,
)


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
    defaults = form_defaults(source_form, objective_sense)
    if defaults is None:
        relations = parse_relations(relation_text, len(matrix))
        domains = parse_variable_domains(domain_text, len(objective))
    else:
        relation, domain = defaults
        relations = (relation,) * len(matrix)
        domains = (domain,) * len(objective)

    constraints = tuple(
        LinearConstraint(row, relation, rhs, label=f"Constraint {index}")
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
