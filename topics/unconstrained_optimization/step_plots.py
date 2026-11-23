"""Method-specific plots drawn from the values recorded for an iteration."""

import numpy as np

from topics.unconstrained_optimization.history import IterationFrame
from topics.unconstrained_optimization.methods.newton_methods import hessian_spectrum
from topics.unconstrained_optimization.models import OptimizationMethod, OptimizationResult
from topics.unconstrained_optimization.visualization import (
    AXES, CONVERGENCE, CURRENT, DIRECTION, NEXT, PATH, PLOT_LIMIT, TRIAL,
    draw_contours, plot_bounds, plot_empty_state, style_axis, style_legend, subplot_bottom,
)


METHOD_TABS = {
    OptimizationMethod.GRADIENT_HALVING: "Line search",
    OptimizationMethod.GRADIENT_PARABOLA: "Parabola fit",
    OptimizationMethod.NEWTON: "Quadratic model",
    OptimizationMethod.BROYDEN: "Broyden update",
}


def plot_method_step(figure, result: OptimizationResult, frame: IterationFrame, selected_trial=None) -> None:
    if result.method is OptimizationMethod.NEWTON:
        _plot_newton_model(figure, result, frame)
    elif result.method is OptimizationMethod.BROYDEN:
        _plot_broyden_update(figure, frame)
    else:
        _plot_line_search(figure, result, frame, selected_trial)


def _plot_line_search(figure, result, frame, selected) -> None:
    trials = frame.trial_points
    if not trials:
        plot_empty_state(figure, "No line-search trials at this point")
        return
    figure.clear()
    figure.subplots_adjust(left=.18, right=.96, bottom=subplot_bottom(figure), top=.88)
    axis = figure.add_subplot()
    plottable = [i for i, t in enumerate(trials) if 0 <= t.step_size < PLOT_LIMIT]
    positive = [trials[i].step_size for i in plottable if trials[i].step_size > 0]
    upper = max(positive, default=1.)
    smallest = min(positive, default=upper)
    if 1e-120 < smallest < upper / 250:
        axis.set_xscale("symlog", linthresh=smallest)
        grid = np.r_[np.linspace(0, smallest, 24), np.geomspace(smallest, upper, 240)]
    else:
        grid = np.linspace(0, upper, 241)
    betas = np.unique(np.r_[grid, [trials[i].step_size for i in plottable]])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinates = [x + betas * d for x, d in zip(frame.current.point, frame.direction)]
        values = result.function.evaluate_grid(*coordinates)
    axis.plot(betas, _finite(values), color=DIRECTION, linewidth=1.45, label="Actual φ(β)")

    search = frame.parabola
    if search and search.interpolation:
        fitted = search.interpolation
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            axis.plot(betas, _finite(fitted.evaluate(betas)), "--", color=CONVERGENCE,
                      linewidth=1.65, label="Interpolating parabola")
            star = search.beta_star
            if star is not None and 0 < star < PLOT_LIMIT and star <= upper:
                axis.axvline(star, color=CONVERGENCE, linestyle=":", linewidth=1.1, label="β*")
                prediction = float(fitted.evaluate(np.array(star)))
                if np.isfinite(prediction) and abs(prediction) < PLOT_LIMIT:
                    axis.scatter(star, prediction, color=CONVERGENCE, marker="x", zorder=6)
    if not search:
        axis.scatter(0, frame.current.value, color=CURRENT, s=21, label="φ(0)", zorder=4)
    _mark_trials(axis, frame, plottable, selected)
    axis.set_xlim(0, upper * 1.08)
    if len(plottable) < len(trials):
        visible = set(plottable)
        outside = [i for i in range(len(trials)) if i not in visible]
        steps = np.linspace(.10, .34, len(outside))
        edge_marks = axis.scatter(np.full(len(outside), .97), steps,
                                  transform=axis.transAxes, c=TRIAL, marker=">", s=26,
                                  label="β outside scale", picker=6, zorder=7)
        edge_marks.set_gid("trials:" + ",".join(map(str, outside)))
        if selected in outside:
            axis.scatter(.97, steps[outside.index(selected)], transform=axis.transAxes,
                         facecolors="none", edgecolors=PATH, marker="o", s=105, zorder=8)
    title = "Objective along −∇f(xₖ)" if not search else "Objective and fitted parabola"
    axis.set_title(title, color=CURRENT, fontsize=9)
    style_axis(axis, "Step size β", "φ(β)")
    style_legend(axis)


def _mark_trials(axis, frame, plottable, selected) -> None:
    trials = frame.trial_points
    groups: dict[tuple[str, str, str, bool], list[int]] = {}
    for i in plottable:
        trial = trials[i]
        valid = trial.value is not None and abs(trial.value) < PLOT_LIMIT
        if not valid:
            style = ("Invalid or off-scale value", TRIAL, "x", False)
        elif i == frame.chosen_trial:
            style = ("Selected β" if frame.parabola else "Accepted", NEXT, "o", True)
        elif frame.parabola and i < 3:
            style = ("Support points", CURRENT, "s", True)
        else:
            style = ("Rejected", TRIAL, "x", True)
        groups.setdefault(style, []).append(i)
    for (label, color, symbol, valid), indices in groups.items():
        options = {} if valid else {"transform": axis.get_xaxis_transform()}
        dots = axis.scatter([trials[i].step_size for i in indices],
                            [trials[i].value if valid else .055 for i in indices],
                            c=color, marker=symbol, s=29, linewidths=1.2, zorder=7,
                            label=label, picker=6, **options)
        dots.set_gid("trials:" + ",".join(map(str, indices)))
    if selected is not None and selected in plottable:
        trial = trials[selected]
        valid = trial.value is not None and abs(trial.value) < PLOT_LIMIT
        options = {} if valid else {"transform": axis.get_xaxis_transform()}
        axis.scatter(trial.step_size, trial.value if valid else .055, marker="o",
                     facecolors="none", edgecolors="#fbbf24", s=110, linewidths=1.4, zorder=8, **options)


def _finite(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values) & (np.abs(values) < PLOT_LIMIT), values, np.nan)


def quadratic_model(offsets, value, gradient, hessian):
    """Evaluate the Newton model on offsets from the recorded base point."""

    p = np.asarray(offsets, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        model = value + p @ gradient + .5 * np.einsum("...i,ij,...j->...", p, hessian, p)
    return _finite(model)


def _plot_newton_model(figure, result, frame) -> None:
    details = frame.second_order
    matrix = details.matrix if details else result.stationary_hessian
    if matrix is None:
        plot_empty_state(figure, "The quadratic model needs a finite Hessian")
        return
    figure.clear()
    figure.subplots_adjust(left=.18, right=.95, bottom=subplot_bottom(figure), top=.86)
    axis = figure.add_subplot()
    value, gradient = frame.current.value, frame.current.gradient
    direction = details.direction if details else None
    try:
        _, curvature = hessian_spectrum(matrix)
    except (ValueError, ArithmeticError):
        curvature = "curvature unavailable"
    minimum = curvature == "positive definite"
    dimension = result.function.dimension
    if dimension == 2:
        xlim, ylim = [plot_bounds([0, d], padding=.40) for d in direction] if direction else ((-1, 1), (-1, 1))
        x, y = np.meshgrid(np.linspace(*xlim, 91), np.linspace(*ylim, 91))
        draw_contours(axis, x, y, quadratic_model(np.stack((x, y), axis=-1), value, gradient, matrix))
        style_axis(axis, "Δ" + result.function.variable_names[0], "Δ" + result.function.variable_names[1])
        axis.set(xlim=xlim, ylim=ylim)
        axis.scatter(0, 0, c=CURRENT, s=34, label="Current point", zorder=5)
        if direction:
            axis.annotate("", direction, xytext=(0, 0),
                          arrowprops={"arrowstyle": "->", "color": NEXT, "linewidth": 1.5}, zorder=5)
            axis.scatter(*direction, c=NEXT, edgecolors=AXES, marker="D", s=53, zorder=6,
                         label="Newton step · model minimum" if minimum else "Newton step")
        elif minimum and not any(gradient):
            axis.scatter(0, 0, facecolors="none", edgecolors=NEXT, s=70, label="Model minimum", zorder=6)
    else:
        _quadratic_slice(axis, result, frame, matrix, direction, minimum)
    axis.set_title(f"Quadratic model · {curvature}", color=CURRENT, fontsize=9)
    style_legend(axis, columns=1)


def _quadratic_slice(axis, result, frame, matrix, direction, minimum) -> None:
    gradient, value = frame.current.gradient, frame.current.value
    dimension = result.function.dimension
    along_direction = dimension > 1 and direction is not None
    if along_direction:
        basis = np.asarray(direction)
        step = frame.beta
        x_label = "β along the full Newton direction"
    else:
        basis = np.eye(1, dimension)[0]
        step = direction[0] if direction is not None else None
        x_label = "Δ" + result.function.variable_names[0]
        if dimension > 1:
            x_label += " (other offsets fixed)"
    xlim = plot_bounds((0, step), padding=.40) if step is not None else (-1, 1)
    scale = np.linspace(*xlim, 241)
    model = quadratic_model(scale[:, None] * basis, value, gradient, matrix)
    axis.plot(scale, model, color=DIRECTION, label="q(p)")
    axis.scatter(0, value, c=CURRENT, s=34, zorder=5, label="Current point")
    if step is not None:
        axis.scatter(step, quadratic_model(step * basis, value, gradient, matrix),
                     c=NEXT, marker="D", s=52, zorder=6,
                     label="Newton step · model minimum" if minimum else "Newton step")
    elif minimum and not any(gradient):
        axis.scatter(0, value, facecolors="none", edgecolors=NEXT, s=74, zorder=6, label="Model minimum")
    axis.set_xlim(*xlim)
    style_axis(axis, x_label, "Model value")


def _plot_broyden_update(figure, frame) -> None:
    details = frame.second_order
    if details is None or details.matrix is None:
        plot_empty_state(figure, "No inverse-Hessian estimate available at this point")
        return
    figure.clear()
    figure.subplots_adjust(left=.17, right=.96, bottom=subplot_bottom(figure), top=.86, hspace=.69, wspace=.29)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.3, 1))
    before, after = (figure.add_subplot(grid[0, i]) for i in range(2))
    updated = details.update.matrix if details.update else None
    values = [*details.matrix, *updated] if updated else details.matrix
    scale = max((abs(x) for row in values for x in row), default=1.) or 1.
    _show_matrix(before, details.matrix, "Bₖ used", scale)
    if updated is not None:
        _show_matrix(after, updated, "Bₖ₊₁", scale)
    else:
        after.set_axis_off()
        reason = details.update_note or "The step did not complete."
        after.text(.5, .54, reason, color="#cbd5e1", ha="center", va="center",
                   fontsize=8, wrap=True, transform=after.transAxes)
    residual = figure.add_subplot(grid[1, :])
    initial = frame.current.gradient
    final = frame.following.gradient if frame.following else None
    components = np.arange(1, len(initial) + 1)
    residual.plot(components, initial, "o-", color=DIRECTION, markersize=3, label="∇f(xₖ)")
    if final is not None:
        residual.plot(components, final, "o-", color=NEXT, markersize=3, label="∇f(xₖ₊₁)")
    if len(initial) <= 10:
        residual.set_xticks(components)
    style_axis(residual, "Component", "Residual")
    style_legend(residual)


def _show_matrix(axis, matrix, title, scale) -> None:
    values = np.asarray(matrix, dtype=float)
    colors = values / scale
    axis.imshow(colors, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    axis.set_title(title, color=CURRENT, fontsize=8)
    dimension = len(values)
    axis.set_xticks([])
    axis.set_yticks([])
    if dimension <= 4:
        for row, column in np.ndindex(values.shape):
            color = "white" if abs(colors[row, column]) > .67 else AXES
            axis.text(column, row, f"{values[row, column]:.3g}", ha="center", va="center",
                      color=color, fontsize=7)


def plot_convergence(figure, result: OptimizationResult, frame: IterationFrame) -> None:
    figure.clear()
    figure.subplots_adjust(left=.21, right=.96, bottom=subplot_bottom(figure), top=.94, hspace=.19)
    objective, gradient = figure.subplots(2, 1, sharex=True)
    iterations = [step.iteration for step in result.steps]
    values = _finite([step.value for step in result.steps])
    norms = _finite([step.gradient_norm if step.gradient_norm is not None else np.nan for step in result.steps])
    for axis, series, label in ((objective, values, "f(x)"), (gradient, norms, "‖∇f‖₂")):
        axis.plot(iterations, series, "o-", color=DIRECTION, markersize=3, linewidth=1.05)
        start = frame.current.iteration
        end = frame.following.iteration if frame.following else start
        axis.axvspan(start - .11, end + .11, color=NEXT, alpha=.14)
        if result.convergence_iteration is not None:
            i = result.convergence_iteration
            axis.scatter(i, series[i], c=CONVERGENCE, marker="D", s=30, zorder=6)
        style_axis(axis, "" if axis is objective else "Iteration k", label)
    tolerance = result.settings.tolerance
    if tolerance < PLOT_LIMIT:
        gradient.set_yscale("symlog", linthresh=max(tolerance, 1e-120))
        gradient.axhline(tolerance, color=PATH, linestyle="--", linewidth=.8, label=f"ε = {tolerance:g}")
        gradient.set_ylim(bottom=0)
        style_legend(gradient)
    gradient.set_xlim(-.2, max(iterations[-1], 1) + .2)
    gradient.xaxis.get_major_locator().set_params(integer=True)
