"""Convert linear programs between the course's four forms.

The four representations implemented here follow Chapter 3.1 of the course
slides: General Form, Canonical Form, Standard Form, and Inequality Form.
The module deliberately has no Qt dependencies so it can be used independently
of the graphical user interface.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from topics.linear_programming.models import (
    ConstraintRelation,
    ConversionResult,
    LinearConstraint,
    LinearProgram,
    ObjectiveSense,
    ProblemForm,
    VariableDomain,
)


def form_defaults(
    form: ProblemForm, sense: ObjectiveSense
) -> tuple[ConstraintRelation, VariableDomain] | None:
    """Return the fixed relation and domain for an input form."""

    if form is ProblemForm.GENERAL:
        return None
    if form is ProblemForm.STANDARD:
        return ConstraintRelation.EQUAL, VariableDomain.NONNEGATIVE
    domain = VariableDomain.FREE if form is ProblemForm.INEQUALITY else VariableDomain.NONNEGATIVE
    return _objective_relation(sense), domain


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

    columns: list[tuple[int, int]] = []
    names: list[str] = []
    for index, (name, domain) in enumerate(zip(problem.variable_names, problem.variable_domains)):
        if domain is VariableDomain.NONNEGATIVE:
            names.append(name)
            columns.append((index, 1))
        elif domain is VariableDomain.NONPOSITIVE:
            names.append(f"{name}_neg")
            columns.append((index, -1))
        else:
            names.extend((f"{name}_plus", f"{name}_minus"))
            columns.extend(((index, 1), (index, -1)))

    objective = _substitute_coefficients(problem.objective, columns)
    constraints = tuple(
        LinearConstraint(
            _substitute_coefficients(constraint.coefficients, columns),
            constraint.relation,
            constraint.right_hand_side,
            constraint.label,
        )
        for constraint in problem.constraints
    )
    return objective, tuple(names), constraints


def _substitute_coefficients(
    coefficients: tuple[Fraction, ...], columns: Sequence[tuple[int, int]]
) -> tuple[Fraction, ...]:
    return tuple(
        coefficients[index] if sign == 1 else -coefficients[index]
        for index, sign in columns
    )


def _to_standard_form(problem: LinearProgram) -> LinearProgram:
    objective, names, base_constraints = _substitute_variables(problem)

    extra_names, constraints = _add_slack_and_surplus_variables(base_constraints)
    output_names = names + extra_names
    output_objective = objective + (Fraction(0),) * len(extra_names)

    return LinearProgram(
        problem.objective_sense,
        output_objective,
        constraints,
        output_names,
        (VariableDomain.NONNEGATIVE,) * len(output_names),
    )


def _add_slack_and_surplus_variables(
    constraints: tuple[LinearConstraint, ...],
) -> tuple[tuple[str, ...], tuple[LinearConstraint, ...]]:
    """Allocate one extra column per inequality and convert the rows to equalities."""

    extra_names: list[str] = []
    extra_columns: dict[int, tuple[int, Fraction]] = {}
    for index, constraint in enumerate(constraints):
        if constraint.relation is ConstraintRelation.EQUAL:
            continue
        slack = constraint.relation is ConstraintRelation.LESS_EQUAL
        extra_columns[index] = (len(extra_names), Fraction(1 if slack else -1))
        prefix = "s" if slack else "u"
        extra_names.append(f"{prefix}_{index + 1}")

    output: list[LinearConstraint] = []
    zero_columns = [Fraction(0)] * len(extra_names)
    for index, constraint in enumerate(constraints):
        extra_coefficients = zero_columns.copy()
        if index in extra_columns:
            column, coefficient = extra_columns[index]
            extra_coefficients[column] = coefficient
        output.append(
            LinearConstraint(
                constraint.coefficients + tuple(extra_coefficients),
                ConstraintRelation.EQUAL,
                constraint.right_hand_side,
                constraint.label,
            )
        )
    return tuple(extra_names), tuple(output)


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
