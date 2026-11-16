"""Solvers for unconstrained continuous optimization.

The algorithms follow the methods introduced in Lectures 8 and 9:
successive-halving gradient descent, direct parabola fitting from a chosen
starting beta, Newton's method, and Broyden's quasi-Newton method.  The module contains no Qt code so
the calculations and iteration history can be tested independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from html import escape as html_escape
from typing import Callable, Sequence

import numpy as np
import sympy as sp


class OptimizationMethod(str, Enum):
    """Available methods from the continuous-optimization lectures."""

    GRADIENT_HALVING = "gradient_halving"
    GRADIENT_PARABOLA = "gradient_parabola"
    NEWTON = "newton"
    BROYDEN = "broyden"

    @property
    def label(self) -> str:
        return {
            OptimizationMethod.GRADIENT_HALVING: "Gradient descent - successive halving",
            OptimizationMethod.GRADIENT_PARABOLA: "Gradient descent - direct parabola fitting",
            OptimizationMethod.NEWTON: "Newton's method",
            OptimizationMethod.BROYDEN: "Broyden's method",
        }[self]


@dataclass(frozen=True)
class FunctionDefinition:
    """Reusable lecture and exercise example kept for future topic extensions."""

    identifier: str
    label: str
    expression: str
    variables: tuple[str, ...]
    default_start: tuple[float, ...]


FUNCTION_LIBRARY: tuple[FunctionDefinition, ...] = (
    FunctionDefinition(
        identifier="bazaraa_shetty",
        label="Bazaraa-Shetty function (Lecture 8 / Exercise 8)",
        expression="(x1 - 2)**4 + (x1 - 2*x2)**2",
        variables=("x1", "x2"),
        default_start=(0.0, 0.0),
    ),
    FunctionDefinition(
        identifier="quadratic_bowl_2d",
        label="Quadratic bowl (Exercise 8)",
        expression="x1**2 + x2**2",
        variables=("x1", "x2"),
        default_start=(3.0, 1.0),
    ),
    FunctionDefinition(
        identifier="quadratic_bowl_3d",
        label="Quadratic bowl in 3D (Exercise 8)",
        expression="x1**2 + x2**2 + x3**2",
        variables=("x1", "x2", "x3"),
        default_start=(3.0, 1.0, -2.0),
    ),
    FunctionDefinition(
        identifier="exercise_9_function",
        label="Exercise 9 function",
        expression="(x1**2 - 2*x1*x2 + x1)**2",
        variables=("x1", "x2"),
        default_start=(2.0, 2.0),
    ),
    FunctionDefinition(
        identifier="quartic_bowl",
        label="Quartic bowl (Exercise 9)",
        expression="x1**4 + x2**4",
        variables=("x1", "x2"),
        default_start=(1.0, 1.0),
    ),
)


@dataclass(frozen=True)
class OptimizationSettings:
    """Numerical settings shared by all algorithms."""

    initial_step: float = 1.0
    tolerance: float = 1e-6
    max_iterations: int = 30

    def __post_init__(self) -> None:
        if self.initial_step <= 0:
            raise ValueError("Initial beta must be greater than zero.")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be greater than zero.")
        if self.max_iterations < 1:
            raise ValueError("Maximum iterations must be at least one.")


@dataclass(frozen=True)
class ParsedFunction:
    """SymPy representation and numerical derivatives of one function."""

    expression: sp.Expr
    symbols: tuple[sp.Symbol, ...]
    gradient_expression: tuple[sp.Expr, ...]
    hessian_expression: sp.Matrix
    _function: Callable[..., object]
    _gradient_function: Callable[..., object]
    _hessian_function: Callable[..., object]

    @property
    def dimension(self) -> int:
        return len(self.symbols)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(str(symbol) for symbol in self.symbols)

    def evaluate(self, point: Sequence[float]) -> float:
        """Evaluate f at a numerical point."""

        values = _validate_point(point, self.dimension)
        result = np.asarray(self._function(*values), dtype=float)
        if result.size != 1 or not np.isfinite(result.reshape(-1)[0]):
            raise ValueError(f"Function is not finite at point {_format_point(values)}.")
        return float(result.reshape(-1)[0])

    def gradient_at(self, point: Sequence[float]) -> np.ndarray:
        """Evaluate the analytic gradient at a numerical point."""

        values = _validate_point(point, self.dimension)
        result = np.asarray(self._gradient_function(*values), dtype=float).reshape(-1)
        if result.size != self.dimension or not np.all(np.isfinite(result)):
            raise ValueError(f"Gradient is not finite at point {_format_point(values)}.")
        return result

    def hessian_at(self, point: Sequence[float]) -> np.ndarray:
        """Evaluate the analytic Hessian at a numerical point."""

        values = _validate_point(point, self.dimension)
        result = np.asarray(self._hessian_function(*values), dtype=float)
        result = result.reshape((self.dimension, self.dimension))
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Hessian is not finite at point {_format_point(values)}.")
        return result


@dataclass(frozen=True)
class LineSearchTrial:
    """One candidate checked while choosing a gradient-descent step size."""

    step_size: float
    point: tuple[float, ...]
    value: float | None
    accepted: bool
    phase: str


@dataclass(frozen=True)
class IterationStep:
    """All values recorded for one main optimization iteration."""

    iteration: int
    point: tuple[float, ...]
    value: float
    gradient: tuple[float, ...]
    gradient_norm: float
    step_size: float
    direction: tuple[float, ...]
    note: str
    trials: tuple[LineSearchTrial, ...] = ()


@dataclass(frozen=True)
class OptimizationResult:
    """Optimization result with the complete intermediate history."""

    function: ParsedFunction
    method: OptimizationMethod
    steps: tuple[IterationStep, ...]
    converged: bool
    message: str

    @property
    def final_point(self) -> tuple[float, ...]:
        return self.steps[-1].point

    @property
    def final_value(self) -> float:
        return self.steps[-1].value


def parse_function(expression_text: str, variable_text: str) -> ParsedFunction:
    """Parse a user expression and create analytic first/second derivatives."""

    variable_names = tuple(
        token for token in re.split(r"[,;\s]+", variable_text.strip()) if token
    )
    if not variable_names:
        raise ValueError("Enter at least one variable, for example x1, x2.")
    if any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in variable_names):
        raise ValueError("Variable names may contain letters, numbers, and underscores only.")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Variable names must be unique.")
    if not expression_text.strip():
        raise ValueError("Enter a function expression.")

    symbols = tuple(sp.Symbol(name, real=True) for name in variable_names)
    local_symbols = dict(zip(variable_names, symbols))
    try:
        expression = sp.sympify(expression_text.replace("^", "**"), locals=local_symbols)
    except (sp.SympifyError, TypeError, ValueError) as error:
        raise ValueError(f"Could not parse the function: {error}") from error

    unknown_symbols = expression.free_symbols.difference(symbols)
    if unknown_symbols:
        unknown = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        raise ValueError(f"Unknown symbols in the function: {unknown}.")
    if expression.is_real is False:
        raise ValueError("The function must be real-valued.")

    gradient = tuple(sp.diff(expression, symbol) for symbol in symbols)
    hessian = sp.hessian(expression, symbols)
    try:
        function = sp.lambdify(symbols, expression, modules="numpy")
        gradient_function = sp.lambdify(symbols, gradient, modules="numpy")
        hessian_function = sp.lambdify(symbols, hessian, modules="numpy")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Could not create numerical derivatives: {error}") from error

    return ParsedFunction(
        expression=expression,
        symbols=symbols,
        gradient_expression=gradient,
        hessian_expression=hessian,
        _function=function,
        _gradient_function=gradient_function,
        _hessian_function=hessian_function,
    )


def optimize(
    function: ParsedFunction,
    method: OptimizationMethod | str,
    start: Sequence[float],
    settings: OptimizationSettings | None = None,
) -> OptimizationResult:
    """Run one selected unconstrained optimization method."""

    method = OptimizationMethod(method)
    settings = settings or OptimizationSettings()
    point = _validate_point(start, function.dimension)
    current_value = function.evaluate(point)
    gradient = function.gradient_at(point)
    steps: list[IterationStep] = [
        IterationStep(
            iteration=0,
            point=tuple(point),
            value=current_value,
            gradient=tuple(gradient),
            gradient_norm=float(np.linalg.norm(gradient)),
            step_size=0.0,
            direction=tuple(np.zeros(function.dimension)),
            note="Initial point",
        )
    ]

    if steps[0].gradient_norm <= settings.tolerance:
        return OptimizationResult(function, method, tuple(steps), True, "Initial point is stationary.")

    inverse_hessian: np.ndarray | None = None
    if method is OptimizationMethod.BROYDEN:
        inverse_hessian = _initial_inverse_hessian(function, point)

    converged = False
    message = "Maximum number of iterations reached."
    for iteration in range(1, settings.max_iterations + 1):
        try:
            if method is OptimizationMethod.GRADIENT_HALVING:
                next_point, step_size, direction, trials, note = _gradient_halving_step(
                    function, point, gradient, settings.initial_step
                )
            elif method is OptimizationMethod.GRADIENT_PARABOLA:
                next_point, step_size, direction, trials, note = _gradient_parabola_step(
                    function, point, gradient, settings.initial_step
                )
            elif method is OptimizationMethod.NEWTON:
                direction, note = _newton_direction(function, point, gradient)
                next_point = point + direction
                step_size = float(np.linalg.norm(direction))
                trials = ()
            else:
                direction, note = _broyden_direction(inverse_hessian, gradient)
                next_point = point + direction
                step_size = float(np.linalg.norm(direction))
                trials = ()

            next_value = function.evaluate(next_point)
            next_gradient = function.gradient_at(next_point)
        except (ValueError, np.linalg.LinAlgError) as error:
            message = f"Stopped at iteration {iteration}: {error}"
            break

        if method is OptimizationMethod.BROYDEN and inverse_hessian is not None:
            inverse_hessian, update_note = _update_inverse_hessian(
                inverse_hessian,
                next_point - point,
                next_gradient - gradient,
            )
            if update_note:
                note = f"{note} {update_note}"

        next_step = IterationStep(
            iteration=iteration,
            point=tuple(next_point),
            value=next_value,
            gradient=tuple(next_gradient),
            gradient_norm=float(np.linalg.norm(next_gradient)),
            step_size=step_size,
            direction=tuple(direction),
            note=note,
            trials=tuple(trials),
        )
        steps.append(next_step)

        if next_step.gradient_norm <= settings.tolerance:
            converged = True
            message = "Converged: gradient norm is below the tolerance."
            break
        if np.linalg.norm(next_point - point) <= settings.tolerance:
            converged = True
            message = "Converged: the point change is below the tolerance."
            break
        if np.allclose(next_point, point):
            message = "Stopped because no improving step was found."
            break

        point = next_point
        current_value = next_value
        gradient = next_gradient

    return OptimizationResult(function, method, tuple(steps), converged, message)


def format_optimization_details(result: OptimizationResult) -> str:
    """Format derivative information and all iteration details for the GUI."""

    gradient_text = ", ".join(str(item) for item in result.function.gradient_expression)
    lines = [
        f"Function: {sp.sstr(result.function.expression)}",
        f"Gradient: ({gradient_text})",
        f"Hessian:\n{result.function.hessian_expression}",
        f"Method: {result.method.label}",
        f"Status: {result.message}",
        f"Final point: {_format_point(result.final_point)}",
        f"Final value: {result.final_value:.10g}",
        "",
        "Iteration details:",
    ]
    for step in result.steps:
        lines.append(
            f"k={step.iteration}: x={_format_point(step.point)}, "
            f"f(x)={step.value:.10g}, ||gradient||={step.gradient_norm:.6g}, "
            f"β={step.step_size:.6g}"
        )
        lines.append(f"  gradient={_format_point(step.gradient)}; {step.note}")
        for trial in step.trials:
            trial_value = "not finite" if trial.value is None else f"{trial.value:.10g}"
            accepted = "accepted" if trial.accepted else "rejected"
            lines.append(
                f"  trial ({trial.phase}): β={trial.step_size:.8g}, "
                f"x={_format_point(trial.point)}, f(x)={trial_value} -> {accepted}"
            )
    return "\n".join(lines)


def format_optimization_details_html(result: OptimizationResult) -> str:
    """Render derivative and iteration data in a LaTeX-like rich-text layout."""

    gradient = sp.Tuple(*result.function.gradient_expression)
    hessian = result.function.hessian_expression
    iteration_rows = "".join(
        "<tr>"
        f"<td>{step.iteration}</td>"
        f"<td>{_html_point(result.function, step.point)}</td>"
        f"<td>{step.value:.10g}</td>"
        f"<td>{step.gradient_norm:.6g}</td>"
        f"<td>{step.step_size:.6g}</td>"
        f"<td>{html_escape(step.note)}</td>"
        "</tr>"
        for step in result.steps
    )
    trials_html = _format_line_search_sections_html(result)
    return f"""<html>
<head>
<style>
body {{ color: #dbeafe; background-color: #0f172a; font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 10pt; }}
.heading {{ color: #f8fafc; font-size: 12pt; font-weight: bold; margin-bottom: 6px; }}
.subheading {{ color: #93c5fd; font-weight: bold; margin-top: 10px; margin-bottom: 4px; }}
.math {{ color: #f8fafc; font-size: 11pt; margin: 4px 0; }}
.math-block {{ color: #f8fafc; font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 10pt; margin: 5px 0 0 0; padding: 8px 10px; background-color: #0b1323; border: 1px solid #263753; border-radius: 5px; white-space: pre-wrap; }}
.formula-card {{ background-color: #111c2f; border: 1px solid #2b3952; border-radius: 6px; padding: 8px 10px; margin: 0 0 8px 0; }}
.formula-title {{ color: #93c5fd; font-weight: bold; margin-bottom: 3px; }}
.formula-symbol {{ color: #f8fafc; font-size: 11pt; }}
.status {{ color: #bbf7d0; background-color: #193b2a; padding: 4px; }}
.summary {{ border-collapse: collapse; margin-top: 4px; }}
.summary td {{ padding: 3px 16px 3px 0; vertical-align: top; }}
.summary .label {{ color: #93c5fd; font-weight: bold; }}
.data {{ border-collapse: collapse; width: 100%; }}
.data td, .data th {{ border-bottom: 1px solid #334155; padding: 4px 5px; text-align: left; vertical-align: top; }}
.data th {{ color: #93c5fd; font-weight: bold; }}
.equation-card {{ background-color: #111c2f; border: 1px solid #2b3952; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }}
.candidate-table {{ table-layout: fixed; }}
.candidate-table td, .candidate-table th {{ padding: 8px 8px; vertical-align: middle; line-height: 1.25; }}
.candidate-table th:nth-child(1), .candidate-table td:nth-child(1) {{ width: 12%; }}
.candidate-table th:nth-child(2), .candidate-table td:nth-child(2) {{ width: 43%; }}
.candidate-table th:nth-child(3), .candidate-table td:nth-child(3) {{ width: 25%; }}
.candidate-table th:nth-child(4), .candidate-table td:nth-child(4) {{ width: 20%; }}
.calculation {{ white-space: normal; }}
.vector {{ display: inline-block; vertical-align: middle; line-height: 1.0; white-space: nowrap; margin: 0 2px; }}
.selected-row td {{ background-color: #193b2a; }}
.done {{ color: #86efac; font-weight: bold; }}
.muted {{ color: #94a3b8; }}
</style>
</head>
<body>
<div class='heading'>Function and derivatives</div>
<div class='formula-card'>
<div class='formula-title'>Objective function</div>
<div class='formula-symbol'><i>f</i>(<i>x</i>) =</div>
<pre class='math-block'>{html_escape(sp.pretty(result.function.expression, use_unicode=True))}</pre>
</div>
<div class='formula-card'>
<div class='formula-title'>Gradient</div>
<div class='formula-symbol'>∇<i>f</i>(<i>x</i>) =</div>
<pre class='math-block'>{html_escape(sp.pretty(gradient, use_unicode=True))}</pre>
</div>
<div class='formula-card'>
<div class='formula-title'>Hessian</div>
<div class='formula-symbol'><i>H</i><sub>f</sub>(<i>x</i>) =</div>
<pre class='math-block'>{html_escape(sp.pretty(hessian, use_unicode=True))}</pre>
</div>
<div class='status'>{html_escape(result.message)}</div>
<div class='subheading'>Optimization summary</div>
<table class='summary'>
<tr><td class='label'>Method</td><td>{html_escape(result.method.label)}</td></tr>
<tr><td class='label'>Final point <i>x</i><sub>*</sub></td><td>{_html_point(result.function, result.final_point)}</td></tr>
<tr><td class='label'><i>f</i>(<i>x</i><sub>*</sub>)</td><td>{result.final_value:.10g}</td></tr>
</table>
<div class='subheading'>Iteration results</div>
<table class='data'>
<tr><th>k</th><th><i>x</i><sub>k</sub></th><th><i>f</i>(<i>x</i><sub>k</sub>)</th><th>‖∇<i>f</i>‖</th><th>β</th><th>Details</th></tr>
{iteration_rows}
</table>
<div class='subheading'>Line-search details</div>
{trials_html}
</body>
</html>"""


def format_line_search_details_html(result: OptimizationResult) -> str:
    """Render only the beta candidate tables for the intermediate-results panel."""

    return f"""<html>
<head>
<style>
body {{ color: #dbeafe; background-color: #0f172a; font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 10pt; }}
.subheading {{ color: #93c5fd; font-weight: bold; font-size: 11pt; margin: 12px 0 6px 0; }}
.math {{ color: #f8fafc; font-size: 11pt; margin: 4px 0; }}
.equation-card {{ background-color: #111c2f; border: 1px solid #2b3952; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }}
.data {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
.data td, .data th {{ border-bottom: 1px solid #334155; padding: 8px 8px; text-align: left; vertical-align: middle; line-height: 1.25; }}
.data th {{ color: #93c5fd; font-weight: bold; }}
.data th:nth-child(1), .data td:nth-child(1) {{ width: 12%; }}
.data th:nth-child(2), .data td:nth-child(2) {{ width: 43%; }}
.data th:nth-child(3), .data td:nth-child(3) {{ width: 25%; }}
.data th:nth-child(4), .data td:nth-child(4) {{ width: 20%; }}
.calculation {{ white-space: normal; }}
.vector {{ display: inline-block; vertical-align: middle; line-height: 1.0; white-space: nowrap; margin: 0 2px; }}
.selected-row td {{ background-color: #193b2a; }}
.done {{ color: #86efac; font-weight: bold; }}
.muted {{ color: #94a3b8; }}
</style>
</head>
<body>
{_format_line_search_sections_html(result)}
</body>
</html>"""


def _html_point(function: ParsedFunction, point: Sequence[float]) -> str:
    values = ", ".join(
        f"{_html_variable_name(name)} = {value:.8g}"
        for name, value in zip(function.variable_names, point)
    )
    return f"({values})"


def _html_variable_name(name: str) -> str:
    match = re.fullmatch(r"([A-Za-z_]+?)(\d+)$", name)
    if match:
        return f"<i>{html_escape(match.group(1))}</i><sub>{match.group(2)}</sub>"
    return f"<i>{html_escape(name)}</i>"


def _format_line_search_sections_html(result: OptimizationResult) -> str:
    """Render the beta candidates in the same calculation style as Exercise 8."""

    sections: list[str] = []
    for step in result.steps:
        if not step.trials:
            continue

        base_step = result.steps[max(step.iteration - 1, 0)]
        trial_rows: list[str] = []
        for trial in step.trials:
            trial_value = "not finite" if trial.value is None else f"{trial.value:.10g}"
            selected = (
                trial.accepted
                and step.step_size > 0
                and np.isclose(trial.step_size, step.step_size)
                and np.allclose(trial.point, step.point)
            )
            if selected:
                outcome = "<span class='done'>done!</span>"
                trial_value = f"{trial_value} &lt; {base_step.value:.10g}"
            else:
                outcome = "accepted" if trial.accepted else "rejected"
            row_class = " class='selected-row'" if selected else ""
            trial_rows.append(
                f"<tr{row_class}>"
                f"<td>{trial.step_size:.8g}</td>"
                f"<td class='calculation'>{_html_vector(trial.point)}</td>"
                f"<td>{trial_value}</td>"
                f"<td>{outcome}<div class='muted'>{html_escape(trial.phase)}</div></td>"
                "</tr>"
            )

        base_iteration = step.iteration - 1
        heading = (
            "Parabola-fitting candidates"
            if result.method is OptimizationMethod.GRADIENT_PARABOLA
            else "Successive halving candidates"
        )
        sections.append(
            f"<div class='subheading'>{heading} for k = {base_iteration}</div>"
            "<div class='equation-card'>"
            "<div class='math'><i>x</i><sub>k+1</sub>(β) = "
            "<i>x</i><sub>k</sub> − β · ∇<i>f</i>(<i>x</i><sub>k</sub>)</div>"
            f"<div class='muted'><i>x</i><sub>{base_iteration}</sub> = "
            f"{_html_vector(base_step.point)}, "
            f"∇<i>f</i>(<i>x</i><sub>{base_iteration}</sub>) = "
            f"{_html_vector(base_step.gradient)}, "
            f"<i>f</i>(<i>x</i><sub>{base_iteration}</sub>) = {base_step.value:.10g}</div>"
            "</div>"
            "<table class='data candidate-table'>"
            "<tr><th>β</th><th>Candidate "
            "<i>x</i><sub>k+1</sub>(β)</th><th>"
            "<i>f</i>(<i>x</i><sub>k+1</sub>(β))</th><th>Result</th></tr>"
            f"{''.join(trial_rows)}</table>"
        )

    return "".join(sections) or "<div class='muted'>No beta line-search trials for this method.</div>"


def _html_vector(values: Sequence[float]) -> str:
    """Format a numeric vector as a compact, LaTeX-like column vector."""

    formatted_values = [f"{float(value):.8g}" for value in values]
    row_count = len(formatted_values)
    if row_count == 1:
        brackets = (("⎡", "⎤"),)
    elif row_count == 2:
        brackets = (("⎡", "⎤"), ("⎣", "⎦"))
    else:
        brackets = tuple(
            ("⎡", "⎤") if index == 0 else
            ("⎣", "⎦") if index == row_count - 1 else
            ("⎢", "⎥")
            for index in range(row_count)
        )
    rows = "<br>".join(
        f"{left}&nbsp;{html_escape(value)}&nbsp;{right}"
        for value, (left, right) in zip(formatted_values, brackets)
    )
    return f"<span class='vector'>{rows}</span>"


def _gradient_halving_step(
    function: ParsedFunction,
    point: np.ndarray,
    gradient: np.ndarray,
    initial_step: float,
) -> tuple[np.ndarray, float, np.ndarray, tuple[LineSearchTrial, ...], str]:
    """Choose beta using Lecture 8's halving/doubling rule.

    The first candidate is always evaluated with the configured starting value
    ``beta_start``.  Only after that evaluation do we replace beta by beta / 2
    or 2 * beta.  Once doubling produces a worse value, the previous candidate
    is retained; it is the point between the current point and the rejected
    oversized step along the search direction.
    """

    base_value = function.evaluate(point)
    direction = -gradient
    trials: list[LineSearchTrial] = []
    beta_start = initial_step
    beta = beta_start

    for _ in range(80):
        candidate = point + beta * direction
        candidate_value = _safe_evaluate(function, candidate)
        accepted = candidate_value is not None and candidate_value < base_value
        phase = "initial beta" if beta == beta_start else "halving"
        trials.append(LineSearchTrial(beta, tuple(candidate), candidate_value, accepted, phase))
        if accepted:
            best_point = candidate
            best_value = candidate_value
            best_beta = beta
            # Rule 1b from Lecture 8: continue doubling while values decrease.
            doubled_beta = beta * 2.0
            for _ in range(40):
                doubled_point = point + doubled_beta * direction
                doubled_value = _safe_evaluate(function, doubled_point)
                doubled_accepted = doubled_value is not None and doubled_value < best_value
                trials.append(
                    LineSearchTrial(
                        doubled_beta,
                        tuple(doubled_point),
                        doubled_value,
                        doubled_accepted,
                        "doubling",
                    )
                )
                if not doubled_accepted:
                    break
                best_point = doubled_point
                best_value = doubled_value
                best_beta = doubled_beta
                doubled_beta *= 2.0
            return (
                best_point,
                best_beta,
                direction,
                tuple(trials),
                "Successive halving/doubling "
                f"(β₀={beta_start:.6g}, β={best_beta:.6g})",
            )
        beta /= 2.0

    return (
        point.copy(),
        0.0,
        direction,
        tuple(trials),
        f"No improving step was found (β₀={beta_start:.6g})",
    )


def _gradient_parabola_step(
    function: ParsedFunction,
    point: np.ndarray,
    gradient: np.ndarray,
    initial_step: float,
) -> tuple[np.ndarray, float, np.ndarray, tuple[LineSearchTrial, ...], str]:
    """Fit a parabola directly from beta_start, without a halving phase."""

    base_value = function.evaluate(point)
    direction = -gradient
    beta_start = initial_step
    beta_point = point + beta_start * direction
    beta_value = _safe_evaluate(function, beta_point)
    double_beta = 2.0 * beta_start
    double_point = point + double_beta * direction
    double_value = _safe_evaluate(function, double_point)
    trials: list[LineSearchTrial] = [
        LineSearchTrial(
            beta_start,
            tuple(beta_point),
            beta_value,
            beta_value is not None and beta_value < base_value,
            "initial beta",
        ),
        LineSearchTrial(
            double_beta,
            tuple(double_point),
            double_value,
            double_value is not None and double_value < base_value,
            "doubling sample",
        ),
    ]

    candidates: list[tuple[float, np.ndarray, float]] = []
    if beta_value is not None and beta_value < base_value:
        candidates.append((beta_start, beta_point, beta_value))
    if double_value is not None and double_value < base_value:
        candidates.append((double_beta, double_point, double_value))

    if beta_value is None or double_value is None:
        if not candidates:
            return (
                point.copy(),
                0.0,
                direction,
                tuple(trials),
                f"Parabola fitting from β₀={beta_start:.6g} (no halving); samples were not finite",
            )
        best_beta, best_point, _ = min(candidates, key=lambda item: item[2])
        return (
            best_point,
            best_beta,
            direction,
            tuple(trials),
            f"Parabola fitting from β₀={beta_start:.6g} (no halving); selected β={best_beta:.6g}",
        )

    denominator = 2.0 * beta_start**2
    a = (double_value - 2.0 * beta_value + base_value) / denominator
    b = (-3.0 * base_value + 4.0 * beta_value - double_value) / (2.0 * beta_start)
    if a > 0 and np.isfinite(a) and np.isfinite(b):
        fitted_beta = -b / (2.0 * a)
        if 0.0 <= fitted_beta <= double_beta:
            fitted_point = point + fitted_beta * direction
            fitted_value = _safe_evaluate(function, fitted_point)
            fitted_accepted = fitted_value is not None and fitted_value < base_value
            trials.append(
                LineSearchTrial(
                    fitted_beta,
                    tuple(fitted_point),
                    fitted_value,
                    fitted_accepted,
                    "parabola fit",
                )
            )
            if fitted_accepted:
                candidates.append((fitted_beta, fitted_point, fitted_value))

    if not candidates:
        return (
            point.copy(),
            0.0,
            direction,
            tuple(trials),
            f"Parabola fitting from β₀={beta_start:.6g} (no halving); no improving point was found",
        )

    best_beta, best_point, _ = min(candidates, key=lambda item: item[2])
    return (
        best_point,
        best_beta,
        direction,
        tuple(trials),
        f"Parabola fitting from β₀={beta_start:.6g} (no halving); selected β={best_beta:.6g}",
    )


def _newton_direction(
    function: ParsedFunction,
    point: np.ndarray,
    gradient: np.ndarray,
) -> tuple[np.ndarray, str]:
    hessian = function.hessian_at(point)
    try:
        direction = -np.linalg.solve(hessian, gradient)
        return direction, "Newton step using the exact Hessian"
    except np.linalg.LinAlgError:
        direction = -np.linalg.pinv(hessian) @ gradient
        return direction, "Newton step using the pseudo-inverse because the Hessian is singular"


def _initial_inverse_hessian(function: ParsedFunction, point: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(function.hessian_at(point))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(function.hessian_at(point))


def _broyden_direction(
    inverse_hessian: np.ndarray | None,
    gradient: np.ndarray,
) -> tuple[np.ndarray, str]:
    if inverse_hessian is None:
        raise ValueError("Broyden's inverse Hessian is not initialized.")
    return -inverse_hessian @ gradient, "Broyden step using the current inverse-Hessian approximation"


def _update_inverse_hessian(
    inverse_hessian: np.ndarray,
    displacement: np.ndarray,
    gradient_change: np.ndarray,
) -> tuple[np.ndarray, str]:
    denominator = float(displacement @ inverse_hessian @ gradient_change)
    if abs(denominator) <= 1e-14:
        return inverse_hessian, "inverse-Hessian update skipped because its denominator is too small"
    correction = np.outer(
        displacement - inverse_hessian @ gradient_change,
        displacement @ inverse_hessian,
    ) / denominator
    return inverse_hessian + correction, "inverse-Hessian approximation updated"


def _safe_evaluate(function: ParsedFunction, point: np.ndarray) -> float | None:
    try:
        return function.evaluate(point)
    except ValueError:
        return None


def _validate_point(point: Sequence[float], dimension: int) -> np.ndarray:
    values = np.asarray(tuple(float(value) for value in point), dtype=float)
    if values.size != dimension:
        raise ValueError(f"Start point must contain exactly {dimension} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Start point must contain finite numbers.")
    return values


def _format_point(point: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):.8g}" for value in point) + ")"
