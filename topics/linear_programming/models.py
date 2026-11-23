"""Data models for linear programs and their converted forms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


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
