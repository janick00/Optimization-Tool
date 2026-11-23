"""HTML representations of optimization runs."""

from __future__ import annotations

import re
from html import escape as html_escape
from typing import Sequence

import sympy as sp

from topics.unconstrained_optimization.methods import ParabolaSearchResult
from topics.unconstrained_optimization.methods.newton_methods import hessian_spectrum
from topics.unconstrained_optimization.history import IterationFrame
from topics.unconstrained_optimization.models import (
    OptimizationMethod,
    OptimizationResult,
)
from topics.unconstrained_optimization.objective import ParsedFunction


def format_optimization_details_html(
    result: OptimizationResult, *, parabola_iteration: int | None = None,
    frame: IterationFrame | None = None, selected_trial: int | None = None,
) -> str:
    """Render the objective, derivatives, and final summary."""

    gradient = sp.Tuple(*result.function.gradient_expression)
    hessian = result.function.hessian_expression
    status_class = "status"
    if result.method is OptimizationMethod.GRADIENT_HALVING:
        point_label = "Last accepted point" if result.accepted_halving_step else "Start point"
        value_label = "<i>f</i>(point)"
        if not result.completed_requested_iterations:
            status_class = "warning"
    elif result.method is OptimizationMethod.GRADIENT_PARABOLA and not result.converged:
        status_class = "warning"
        point_label = "Last accepted point" if len(result.steps) > 1 else "Start point"
        value_label = "<i>f</i>(point)"
    elif result.method in (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN):
        point_label = "Stationary point estimate" if result.converged else "Final point"
        value_label = "<i>f</i>(point)"
        if not result.converged:
            status_class = "warning"
    else:
        point_label = "Final point <i>x</i><sub>*</sub>"
        value_label = "<i>f</i>(<i>x</i><sub>*</sub>)"
    iteration_details = ""
    if frame is not None:
        iteration_details = _format_iteration_math(result, frame, selected_trial)
    elif parabola_iteration is not None:
        base, search = result.parabola_iterations[parabola_iteration - 1]
        iteration_details = _format_parabola_details(base.iteration + 1, search)
    return f"""<html>
<head>
<style>
body {{ color: #dbeafe; background-color: #0f172a; font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 10pt; }}
.heading {{ color: #f8fafc; font-size: 12pt; font-weight: bold; margin-bottom: 6px; }}
.subheading {{ color: #93c5fd; font-weight: bold; margin-top: 10px; margin-bottom: 4px; }}
.math-block {{ color: #f8fafc; font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 10pt; margin: 5px 0 0 0; padding: 8px 10px; background-color: #0b1323; border: 1px solid #263753; border-radius: 5px; white-space: pre-wrap; }}
.formula-card {{ background-color: #111c2f; border: 1px solid #2b3952; border-radius: 6px; padding: 8px 10px; margin: 0 0 8px 0; }}
.formula-title {{ color: #93c5fd; font-weight: bold; margin-bottom: 3px; }}
.formula-symbol {{ color: #f8fafc; font-size: 11pt; }}
.status {{ color: #bbf7d0; background-color: #193b2a; padding: 4px; }}
.warning {{ color: #fde68a; background-color: #493b1a; padding: 4px; }}
.summary {{ border-collapse: collapse; margin-top: 4px; }}
.summary td {{ padding: 3px 16px 3px 0; vertical-align: top; }}
.summary .label {{ color: #93c5fd; font-weight: bold; }}
</style>
</head>
<body>
{iteration_details}
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
<div class='{status_class}'>{html_escape(result.message)}</div>
<div class='subheading'>Optimization summary</div>
<table class='summary'>
<tr><td class='label'>Method</td><td>{html_escape(result.method.label)}</td></tr>
<tr><td class='label'>{point_label}</td><td>{_html_point(result.function, result.final_point)}</td></tr>
<tr><td class='label'>{value_label}</td><td>{result.final_value:.10g}</td></tr>
</table>
</body>
</html>"""


def format_intermediate_details_html(
    result: OptimizationResult, *, frames: tuple[IterationFrame, ...] = (),
    active: int = 0, selected_trial: int | None = None, trials_only: bool = False,
) -> str:
    """Render the calculations shown in the intermediate-results panel."""

    if frames:
        if trials_only:
            content = _format_selected_trials(frames[active], selected_trial)
        else:
            content = _format_playback_table(result, frames, active)
    elif result.method is OptimizationMethod.GRADIENT_HALVING:
        content = _format_halving_trials_html(result)
    elif result.method is OptimizationMethod.GRADIENT_PARABOLA:
        content = _format_parabola_table(result)
    else:
        content = f"""
<div class='subheading'>Iteration results</div>
<table class='data iteration-table'>
<tr><th>k</th><th><i>x</i><sub>k</sub></th><th><i>f</i>(<i>x</i><sub>k</sub>)</th><th>‖∇<i>f</i>‖</th><th>{_step_label(result)}</th><th>Details</th></tr>
{_format_iteration_rows(result)}
</table>
{_format_intermediate_sections_html(result)}
"""
    return f"""<html>
<head>
<style>
body {{ color: #dbeafe; background-color: #0f172a; font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 10pt; }}
.subheading {{ color: #93c5fd; font-weight: bold; font-size: 11pt; margin: 12px 0 6px 0; }}
.math {{ color: #f8fafc; font-size: 11pt; margin: 4px 0; }}
.equation-card {{ background-color: #111c2f; border: 1px solid #2b3952; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }}
.data, .parabola-table {{ border-collapse: collapse; table-layout: fixed; font-size: 10pt; }}
.data {{ width: 100%; }}
.data td, .data th, .parabola-table td, .parabola-table th {{ border-bottom: 1px solid #334155; padding: 4px 5px; text-align: left; vertical-align: middle; line-height: 1.2; word-wrap: break-word; }}
.data th, .parabola-table th {{ color: #93c5fd; font-weight: bold; }}
.iteration-table th:nth-child(1), .iteration-table td:nth-child(1) {{ width: 6%; }}
.iteration-table th:nth-child(2), .iteration-table td:nth-child(2) {{ width: 27%; }}
.iteration-table th:nth-child(3), .iteration-table td:nth-child(3) {{ width: 15%; }}
.iteration-table th:nth-child(4), .iteration-table td:nth-child(4) {{ width: 15%; }}
.iteration-table th:nth-child(5), .iteration-table td:nth-child(5) {{ width: 12%; }}
.iteration-table th:nth-child(6), .iteration-table td:nth-child(6) {{ width: 25%; }}
.iteration-table .note {{ word-wrap: break-word; }}
.halving-table td, .halving-table th, .parabola-table td, .parabola-table th {{ padding: 4px 2px; }}
a {{ color: #93c5fd; }}
.vector {{ display: inline-block; vertical-align: middle; line-height: 1.0; white-space: nowrap; margin: 0 1px; }}
.selected-row td {{ background-color: #193b2a; }}
.done {{ color: #86efac; font-weight: bold; }}
.muted {{ color: #94a3b8; }}
</style>
</head>
<body>
{content}
</body>
</html>"""


def _step_label(result: OptimizationResult) -> str:
    if result.method in (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN):
        return "‖Δx‖"
    return "β"


def _format_iteration_rows(result: OptimizationResult) -> str:
    return "".join(
        "<tr>"
        f"<td>{step.iteration}</td>"
        f"<td>{_html_vector(step.point)}</td>"
        f"<td>{step.value:.8g}</td>"
        f"<td>{step.gradient_norm:.6g}</td>"
        f"<td>{step.step_size:.6g}</td>"
        f"<td class='note'>{html_escape(step.note)}</td>"
        "</tr>"
        for step in result.steps
    )


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


def _format_intermediate_sections_html(result: OptimizationResult) -> str:
    if result.method in (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN):
        return _format_second_order_details(result)
    return ""


def _format_second_order_details(result: OptimizationResult) -> str:
    sections = ["<div class='subheading'>Step details</div>"]
    for current, following in zip(result.steps, result.steps[1:]):
        details = following.second_order
        if details is None:
            continue
        is_broyden = result.method is OptimizationMethod.BROYDEN
        matrix_label = "Inverse Hessian approximation B used" if is_broyden else "Hessian H(xₖ) used"
        matrix = "\n".join(
            "[ " + "  ".join(_number(value, precision=10) for value in row) + " ]"
            for row in details.matrix
        )
        differences = ""
        if details.displacement is not None and details.gradient_change is not None:
            differences = (
                f"<div>d = xₖ₊₁ − xₖ = {_compact_vector(details.displacement)}</div>"
                f"<div>g = ∇f(xₖ₊₁) − ∇f(xₖ) = {_compact_vector(details.gradient_change)}</div>"
            )
        sections.append(
            "<div class='equation-card'>"
            f"<div class='subheading'>Iteration {following.iteration}</div>"
            f"<div>Current point xₖ = {_compact_vector(current.point)}</div>"
            f"<div>Gradient ∇f(xₖ) = {_compact_vector(current.gradient or ())}</div>"
            f"<div>{matrix_label}</div><pre>{matrix}</pre>{differences}"
            f"<div>New point xₖ₊₁ = {_compact_vector(following.point)}</div>"
            f"<div>New function value f(xₖ₊₁) = {_number(following.value, precision=10)}</div>"
            f"<div>New gradient norm ‖∇f(xₖ₊₁)‖ = {_number(following.gradient_norm, precision=10)}</div>"
            "</div>"
        )
    if len(sections) == 1:
        sections.append(f"<div class='muted'>{html_escape(result.message)}</div>")
    return "".join(sections)


def _compact_vector(values: Sequence[float], *, precision: int = 10) -> str:
    return "(" + ", ".join(_number(value, precision=precision) for value in values) + ")"


def _format_halving_trials_html(result: OptimizationResult) -> str:
    start_marker = (
        "<p class='done'>Start point (0) · Convergence reached</p>"
        if result.first_convergence_iteration == 0 else ""
    )
    trial_groups = [(base, step.trials) for base, step in zip(result.steps, result.steps[1:])]
    if result.failed_trials:
        trial_groups.append((result.steps[-1], result.failed_trials))
    if not trial_groups:
        return f"{start_marker}<p>{html_escape(result.message)}</p>"

    rows: list[str] = []
    for base, trials in trial_groups:
        iteration = base.iteration + 1
        point = _html_vector(base.point)
        gradient = _html_vector(base.gradient or ())
        for index, trial in enumerate(trials, start=1):
            value = _number(trial.value, precision=8)
            decision = (
                "<span class='done'>Accepted</span>"
                if trial.accepted else html_escape(trial.rejection_reason or "No improvement")
            )
            if trial.accepted and iteration == result.first_convergence_iteration:
                decision += "<br><span class='done'>Convergence reached</span>"
            row_class = " class='selected-row'" if trial.accepted else ""
            rows.append(
                f"<tr{row_class}><td>{iteration}</td><td>{index}</td><td>{point}</td><td>{gradient}</td>"
                f"<td>{base.value:.8g}</td><td>{trial.step_size:.8g}</td>"
                f"<td>{_html_vector(trial.point)}</td><td>{value}</td><td>{decision}</td></tr>"
            )

    return (
        "<div class='subheading'>Successive halving candidates</div>"
        f"{start_marker}"
        "<div class='math'><i>x</i><sub>test</sub> = "
        "<i>x</i><sub>k</sub> − β · ∇<i>f</i>(<i>x</i><sub>k</sub>)</div>"
        "<table class='data halving-table'>"
        "<tr><th>Iter.</th><th>Trial</th><th>Current point</th><th>Gradient</th>"
        "<th><i>f</i>(<i>x</i><sub>k</sub>)</th><th>β</th><th>Test point</th>"
        "<th><i>f</i>(<i>x</i><sub>test</sub>)</th><th>Decision</th></tr>"
        f"{''.join(rows)}</table>"
    )


def _format_parabola_table(result: OptimizationResult) -> str:
    rows: list[str] = []
    for base, search in result.parabola_iterations:
        index = base.iteration + 1
        start, first, second = search.samples
        selected = search.selected
        step_size = _number(selected.step_size if selected else None)
        new_point = _html_vector(selected.point, precision=6) if selected else "—"
        rows.append(
            f"<tr><td><a href='#parabola-{index}'>{index}</a></td>"
            f"<td>{_html_vector(start.point, precision=6)}</td>"
            f"<td>{_html_vector(base.gradient or (), precision=6)}</td>"
            f"<td>{_number(first.step_size)}</td><td>{_number(start.value)}</td>"
            f"<td>{_number(first.value)}</td><td>{_number(second.value)}</td>"
            f"<td>{_number(search.beta_star)}</td>"
            f"<td>{step_size}</td><td>{new_point}</td></tr>"
        )
    if not rows:
        return f"<p>{html_escape(result.message)}</p>"
    return (
        "<table class='parabola-table' width='530'><tr>"
        "<th>Iteration</th><th>Current point</th><th>Gradient</th><th>β</th>"
        "<th>f₀</th><th>f₁</th><th>f₂</th><th>β*</th><th>Selected β</th><th>New point</th>"
        f"</tr>{''.join(rows)}</table>"
    )


def _format_parabola_details(iteration: int, search: ParabolaSearchResult) -> str:
    start, first, _ = search.samples
    fitted, selected = search.fitted, search.selected
    fit = search.interpolation
    calc = "<p>β* = (β / 2) · (3f₀ − 4f₁ + f₂) / (f₀ − 2f₁ + f₂)</p>"
    if fit is not None:
        f0, f1, f2 = (_number(value, precision=10) for value in (fit.f0, fit.f1, fit.f2))
        calc += (
            f"<p>D = ({f0}) − 2 · ({f1}) + ({f2}) = {_number(fit.denominator, precision=10)}<br>"
            f"3f₀ − 4f₁ + f₂ = 3 · ({f0}) − 4 · ({f1}) + ({f2}) = "
            f"{_number(fit.numerator, precision=10)}</p>"
        )
        calc += (
            f"<p>t = β<sub>test</sub> / {_number(fit.beta, precision=10)}<br>"
            f"q(t) = {f0} + ({f1} − {f0})t + ({_number(fit.denominator, precision=10)} / 2)t(t − 1)<br>"
            "q′(t) = f₁ − f₀ + D(t − ½). Set q′(t) = 0 for the stationary point. "
            "It is a minimum when D &gt; 0.</p>"
        )
        if search.beta_star is not None:
            calc += (
                f"<p>β* = ({fit.beta:.10g} / 2) · "
                f"({fit.numerator:.10g} / {fit.denominator:.10g}) = {search.beta_star:.10g}</p>"
            )

    trials = [(label, trial) for label, trial in zip(("f₀", "f₁ = fA", "f₂"), search.samples)]
    if fitted is not None:
        trials.append(("fB", fitted))
    rows = "".join(
        f"<tr><td>{label}</td><td>{trial.step_size:.10g}</td>"
        f"<td>{_html_vector(trial.point)}</td><td>{_number(trial.value, precision=10)}</td></tr>"
        for label, trial in trials
    )
    comparison = ""
    if first.value is not None and fitted is not None and fitted.value is not None:
        symbol = "=" if first.value == fitted.value else "&lt;" if first.value < fitted.value else "&gt;"
        initial_value, fitted_value = _comparison_values(first.value, fitted.value)
        comparison = f"<p>fA = {initial_value} {symbol} {fitted_value} = fB</p>"
    decision = (
        f"Selected {'β' if selected is first else 'β*'} = {selected.step_size:.10g}."
        f" New point {_html_vector(selected.point)}."
        if selected is not None else html_escape(search.failure_reason or "")
    )
    if selected is not None and start.value is not None and selected.value is not None:
        current_value, new_value = _comparison_values(start.value, selected.value)
        decision += f" The function decreases from {current_value} to {new_value}."
    if selected is first and fitted is not None and first.value == fitted.value:
        decision += " Equal function values favour the initial β."
    issue = f"<p>{html_escape(search.fit_issue)}</p>" if search.fit_issue and selected else ""
    return (
        f"<div class='heading'>Parabola fit · Iteration {iteration}</div>"
        "<div class='formula-card'>"
        "<p>x(β) = xₖ − β · ∇f(xₖ)</p>"
        "<table class='summary' width='100%'>"
        f"<tr><td>Sample</td><td>β</td><td>Test point</td><td>Value</td></tr>{rows}</table>"
        f"{calc}{comparison}{issue}<p>{decision}</p></div>"
    )


def _number(value: float | None, *, precision: int = 6) -> str:
    return "—" if value is None else f"{value:.{precision}g}"


def _comparison_values(first: float, second: float) -> tuple[str, str]:
    if first != second and f"{first:.10g}" == f"{second:.10g}":
        return repr(first), repr(second)
    return f"{first:.10g}", f"{second:.10g}"


def _html_vector(values: Sequence[float], *, precision: int = 8) -> str:
    """Format a numeric vector as a compact, LaTeX-like column vector."""

    formatted_values = [_number(float(value), precision=precision) for value in values]
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


def _format_playback_table(result, frames, active) -> str:
    frame = frames[active]
    start = result.steps[0]
    initial = _compact_vector(start.point)
    convergence = result.convergence_iteration
    start_note = " · Convergence reached" if convergence == 0 else ""
    step_note = html_escape(frame.following.note) if frame.following else html_escape(frame.failure or result.message)
    selected = (
        "<div class='subheading'>Step details</div>"
        f"<div>Current point xₖ = {_compact_vector(frame.current.point)} · f(xₖ) = {_number(frame.current.value)}</div>"
    )
    if frame.following:
        selected += (
            f"<div>New point xₖ₊₁ = {_compact_vector(frame.following.point)}</div>"
            f"<div>New function value f(xₖ₊₁) = {_number(frame.following.value, precision=10)}</div>"
        )
    selected += f"<div class='muted'>{step_note}</div>"
    overview = []
    for index, item in enumerate(frames):
        next_step = item.following
        point = _compact_vector(next_step.point, precision=6) if next_step else "—"
        value = _number(next_step.value if next_step else None)
        norm = _number(next_step.gradient_norm if next_step else None)
        note = "Attempt ended" if item.failure else ""
        if next_step and convergence == next_step.iteration:
            note = "Convergence reached"
        color = " bgcolor='#233c59'" if index == active else ""
        overview.append(
            f"<tr{color}><td>{_table_link(f'iteration-{item.number}', str(item.number))}</td>"
            f"<td>{point}</td><td>{value}</td><td>{norm}</td><td>{_number(item.beta)}</td></tr>"
        )
        if note:
            overview.append(f"<tr{color}><td colspan='5' class='muted'>{note}</td></tr>")
    table = (
        "<div class='subheading'>Iteration results</div>"
        f"<div>Start point x₀ = {initial}{start_note}</div><table class='data'>"
        "<tr><th>Step</th><th>New point</th><th>f(new)</th><th>‖∇f‖</th><th>β</th></tr>"
        + "".join(overview) + "</table>"
    )
    return table + selected


def _format_selected_trials(frame: IterationFrame, selected: int | None) -> str:
    if not frame.trial_points:
        return "<p>No trials at this point.</p>"
    title = "Samples and trials" if frame.parabola else "Successive halving candidates"
    rows = []
    for i, trial in enumerate(frame.trial_points):
        color = " bgcolor='#233c59'" if i == selected else " bgcolor='#193b2a'" if i == frame.chosen_trial else ""
        beta = _number(trial.step_size)
        if frame.parabola:
            beta += " " + ("f₀", "f₁", "f₂", "β*")[i]
        status = frame.trial_decision(i)
        status = {"Reference": "Base", "Not selected": "Skipped"}.get(status, status)
        rows.append(
            f"<tr{color}><td>{_table_link(f'trial-{i}', beta)}</td>"
            f"<td>{_compact_vector(trial.point, precision=6)}</td>"
            f"<td>{_number(trial.value)}</td><td>{status}</td></tr>"
        )
    return (
        f"<div class='subheading'>{title} · {frame.number}</div><table class='data'>"
        "<tr><th>β</th><th>x(β)</th><th>φ(β)</th><th>Status</th></tr>"
        + "".join(rows) + "</table>"
    )


def _table_link(fragment: str, text: str) -> str:
    return f"<a href='#{fragment}' style='color: #93c5fd'>{text}</a>"


def _format_iteration_math(result, frame, selected_trial) -> str:
    current = frame.current
    vector = _compact_vector(current.gradient or ())
    is_halving = result.method is OptimizationMethod.GRADIENT_HALVING
    content = f"<div class='heading'>Iteration {frame.number}</div>"
    if is_halving and selected_trial is not None:
        content += _format_trial_calculation(result, frame, selected_trial)
    content += (
        f"<div class='formula-card'>xₖ = {_compact_vector(current.point)}<br>"
        f"f(xₖ) = {_number(current.value, precision=10)}<br>∇f(xₖ) = {vector}</div>"
    )
    if frame.second_order is not None:
        content += _format_matrix_calculation(result, frame)
    elif frame.trials:
        direction = _compact_vector(frame.direction or ())
        prefix = "dₖ = " if is_halving else ""
        content += f"<p>{prefix}−∇f(xₖ) = −{vector} = {direction}</p>"
        if frame.parabola:
            content += _format_parabola_details(frame.number, frame.parabola)
            if selected_trial is not None:
                content += _format_trial_calculation(result, frame, selected_trial)
        elif current.iteration > 0:
            content += f"<p>Initial trial β = {_number(frame.trials[0].step_size, precision=10)}, the preceding accepted step.</p>"
    elif result.stationary_hessian is not None:
        content += _matrix_block("Hessian at the stationary start", result.stationary_hessian)
        content += _hessian_description(result.stationary_hessian)
    if frame.following:
        step = frame.following
        content += (
            f"<p>xₖ₊₁ = {_compact_vector(current.point)} + {_number(frame.beta, precision=10)} · "
            f"{_compact_vector(frame.direction or ())} = {_compact_vector(step.point)}<br>"
            f"f(xₖ₊₁) = {_number(step.value, precision=10)}</p>"
        )
    if frame.failure:
        content += f"<p class='warning'>{html_escape(frame.failure)}</p>"
    return content + _format_convergence(result, frame)


def _format_trial_calculation(result, frame, index) -> str:
    trial = frame.trial_points[index]
    current = frame.current
    beta = _number(trial.step_size, precision=10)
    is_halving = result.method is OptimizationMethod.GRADIENT_HALVING
    candidate = "x_trial" if is_halving else "x(β)"
    if is_halving:
        decision = "Accepted" if trial.accepted else "Rejected"
        text = (
            f"<div class='subheading'>Selected trial {index + 1} · {decision}</div>"
            f"<p>x_trial = xₖ + β · dₖ<br>"
            f"= {_compact_vector(current.point)} + {beta} · "
            f"{_compact_vector(frame.direction or ())} = {_compact_vector(trial.point)}</p>"
        )
    else:
        text = (
            f"<div class='subheading'>Selected row · {frame.trial_label(index)}</div>"
            f"<p>x(β) = {_compact_vector(current.point)} − {beta} · "
            f"{_compact_vector(current.gradient or ())} = {_compact_vector(trial.point)}</p>"
        )
    if is_halving and index > 0:
        previous = frame.trial_points[index - 1]
        text += (f"<p>Reduced β = {_number(previous.step_size, precision=10)} · {result.settings.halving_factor:g} "
                 f"= {beta}.</p>")
    if trial.value is not None:
        value, base = _comparison_values(trial.value, current.value)
        relation = "&lt;" if trial.value < current.value else "&gt;" if trial.value > current.value else "="
        text += f"<p>φ(β) = f({candidate}) = {value} {relation} {base} = f(xₖ).</p>"
    if is_halving:
        return text + (f"<p>{html_escape(trial.rejection_reason)}.</p>" if trial.rejection_reason else "")
    explanation = frame.trial_decision(index)
    if trial.rejection_reason:
        explanation += f" · {trial.rejection_reason}"
    return text + f"<p>{html_escape(explanation)}.</p>"


def _format_matrix_calculation(result, frame) -> str:
    details = frame.second_order
    matrix = details.matrix
    grad = frame.current.gradient or ()
    direction = details.direction
    if result.method is OptimizationMethod.NEWTON:
        if matrix is None:
            return "<p>The Hessian could not be evaluated at this point.</p>"
        text = _matrix_block("Hessian H(xₖ) used", matrix) + _hessian_description(matrix)
        text += f"<p>Solve H(xₖ) · dₖ = −∇f(xₖ) = {_compact_vector(tuple(-g for g in grad))}.</p>"
        text += (
            f"<p>q(p) = {_number(frame.current.value, precision=10)} + "
            f"{_compact_vector(grad)}ᵀ p + ½ pᵀ H(xₖ) p.</p>"
        )
    else:
        text = ""
        if frame.current.iteration == 0 and result.initial_hessian is not None:
            text = _matrix_block("Initial Hessian H(x₀)", result.initial_hessian)
            text += "<p>Initialize B₀ by solving H(x₀) · B₀ = I.</p>"
        if matrix is not None:
            text += _matrix_block("Inverse Hessian approximation B used", matrix)
            text += f"<p>dₖ = −Bₖ · {_compact_vector(grad)}.</p>"
    if direction is not None:
        text += f"<p>dₖ = {_compact_vector(direction)} · β = {details.beta:g} (full step)</p>"
    if frame.following is None and details.candidate is not None:
        text += f"<p>Trial point = {_compact_vector(details.candidate)}."
        if details.candidate_value is not None:
            text += f" Trial f = {_number(details.candidate_value, precision=10)}."
        text += "</p>"
    if details.gradient_change is not None:
        text += _format_broyden_update(details, frame)
    return text


def _format_broyden_update(details, frame) -> str:
    text = (
        "<div class='subheading'>Inverse Broyden update</div>"
        f"<p>s = xₖ₊₁ − xₖ = {_compact_vector(details.displacement)}<br>"
        f"y = ∇f(xₖ₊₁) − ∇f(xₖ) = {_compact_vector(frame.following.gradient)} − "
        f"{_compact_vector(frame.current.gradient)} = {_compact_vector(details.gradient_change)}</p>"
        "<p>Bₖ₊₁ = Bₖ − (Bₖy − s)(sᵀBₖ) / (sᵀBₖy)</p>"
    )
    update = details.update
    if update is None:
        return text + f"<p class='warning'>{html_escape(details.update_note)}</p>"
    text += (
        f"<p>Scaled calculation with ‖s‖∞ = {_number(update.displacement_scale, precision=10)}, "
        f"‖y‖∞ = {_number(update.gradient_scale, precision=10)} and "
        f"‖(s/‖s‖∞)ᵀBₖ‖∞ = {_number(update.row_scale, precision=10)}.</p>"
        f"<p>r = (s/‖s‖∞)ᵀBₖ / {_number(update.row_scale, precision=10)} = {_compact_vector(update.row)}<br>"
        f"ŷ = y / ‖y‖∞ = {_compact_vector(update.scaled_gradient_change)}<br>"
        f"u = Bₖŷ − s/‖y‖∞ = {_compact_vector(update.discrepancy)}<br>"
        f"r · ŷ = {_number(update.denominator, precision=10)}</p>"
    )
    return text + _matrix_block("Correction u r / (r · ŷ)", update.correction) + _matrix_block(
        "Updated inverse Bₖ₊₁", update.matrix,
    )


def _matrix_block(label, matrix) -> str:
    values = "\n".join("[ " + "   ".join(_number(x, precision=8) for x in row) + " ]" for row in matrix)
    return f"<div class='subheading'>{label}</div><pre class='math-block'>{values}</pre>"


def _hessian_description(matrix) -> str:
    try:
        eigenvalues, classification = hessian_spectrum(matrix)
    except (ValueError, ArithmeticError):
        return "<p>The eigenvalue test is unavailable.</p>"
    message = (
        "The quadratic model has a unique minimum." if classification == "positive definite" else
        "The quadratic model has no unique minimum."
    )
    return f"<p>Eigenvalues {_compact_vector(eigenvalues)} · Hessian {classification}. {message}</p>"


def _format_convergence(result, frame) -> str:
    last = frame.following or frame.current
    norm = last.gradient_norm
    if norm is None:
        return "<p>The new gradient is undefined.</p>"
    tolerance = result.settings.tolerance
    sign = "&lt;" if norm < tolerance else "≥"
    left, right = _comparison_values(norm, tolerance)
    label = "xₖ₊₁" if frame.following else "xₖ"
    reached = " Convergence reached." if result.convergence_iteration == last.iteration else ""
    return f"<p>‖∇f({label})‖ = {left} {sign} ε = {right}.{reached}</p>"
