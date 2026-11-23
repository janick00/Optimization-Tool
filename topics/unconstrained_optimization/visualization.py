"""Shared trajectory plot and Matplotlib styling for the iteration views."""

from __future__ import annotations

from math import isfinite

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from topics.unconstrained_optimization.history import IterationFrame, iteration_history
from topics.unconstrained_optimization.models import OptimizationMethod, OptimizationResult


BACKGROUND = "#111827"
AXES = "#0f172a"
PATH = "#fbbf24"
CURRENT = "#f8fafc"
NEXT = "#34d399"
DIRECTION = "#93c5fd"
TRIAL = "#fb923c"
CONVERGENCE = "#d8b4fe"
PLOT_LIMIT = 1e145

Limits = tuple[tuple[float, float], tuple[float, float]]


def subplot_bottom(figure: Figure) -> float:
    """Keep space for the tick labels and axis title when the canvas shrinks."""

    label_space = 43 * figure.dpi / 100
    return max(.24, min(.48, label_space / max(figure.bbox.height, 1.)))


def plot_empty_state(figure: Figure, message: str = "Run a valid optimization to display the plot") -> None:
    figure.clear()
    axis = figure.add_subplot()
    axis.set_facecolor(AXES)
    axis.text(.5, .5, message, color="#94a3b8", ha="center", va="center", wrap=True,
              transform=axis.transAxes)
    axis.set_axis_off()


def plot_optimization_result(
    figure: Figure, result: OptimizationResult, frame: IterationFrame | None = None, *,
    selected_trial: int | None = None, focus: bool = False, view: Limits | None = None,
) -> Axes:
    """Draw the recorded path and the selected transition, on a local slice if needed."""

    frame = frame or iteration_history(result)[0]
    if figure.axes:
        axis = figure.axes[0]
        axis.clear()
    else:
        axis = figure.add_subplot()
    axis.set_axis_on()
    figure.subplots_adjust(left=.17, right=.96, bottom=subplot_bottom(figure), top=.90)
    one_dimensional = result.function.dimension == 1
    steps = tuple(_plot_point(step.point, step.value, one_dimensional) for step in result.steps)
    current = _plot_point(frame.current.point, frame.current.value, one_dimensional)
    following = (_plot_point(frame.following.point, frame.following.value, one_dimensional)
                 if frame.following else None)
    trial = None
    if selected_trial is not None:
        sample = frame.trial_points[selected_trial]
        trial = _plot_point(sample.point, sample.value, one_dimensional)
    elif frame.failure and frame.second_order and frame.second_order.candidate is not None:
        failed = frame.second_order
        trial = _plot_point(failed.candidate, failed.candidate_value, one_dimensional)
    local = tuple(p for p in (current, following, trial) if p is not None)
    anchors = (*steps, *local) if not focus else local
    xlim, ylim = view or tuple(plot_bounds(column, padding=.24) for column in zip(*anchors))

    function = result.function
    x = np.linspace(*xlim, 240 if one_dimensional else 91)
    if one_dimensional:
        y = function.evaluate_grid(x)
        y = np.where(np.abs(y) < PLOT_LIMIT, y, np.nan)
        axis.plot(x, y, color="#60a5fa", linewidth=1.5, label="f(x)")
        if view is None:
            # Use the ordinate range of the plot as well as the recorded points.
            ordinate = y[np.isfinite(y) & (np.abs(y) < PLOT_LIMIT)]
            if ordinate.size:
                ylim = plot_bounds((*[p[1] for p in anchors], *np.percentile(ordinate, (2, 96))), padding=.10)
        style_axis(axis, function.variable_names[0], "f(x)")
    else:
        y = np.linspace(*ylim, 91)
        grid_x, grid_y = np.meshgrid(x, y)
        draw_contours(axis, grid_x, grid_y, function.evaluate_grid(grid_x, grid_y, *frame.current.point[2:]))
        style_axis(axis, *function.variable_names[:2])
    axis.set(xlim=xlim, ylim=ylim)
    axis.set_title("Trajectory" if function.dimension <= 2 else "2D slice at xₖ · higher coordinates fixed",
                   color=CURRENT, fontsize=9)
    if not focus and len(steps) <= 25:
        for index, point in enumerate(steps):
            if index != frame.current.iteration and (frame.following is None or index != frame.following.iteration):
                axis.annotate(str(index), point, xytext=(4, 4), textcoords="offset points",
                              color="#cbd5e1", fontsize=7, zorder=5, annotation_clip=True)
    outline_trial = result.method is OptimizationMethod.GRADIENT_HALVING
    _plot_path(axis, steps, result.convergence_iteration, frame, current, following, trial, outline_trial)
    if one_dimensional and trial is not None and np.isfinite(trial[0]) and not np.isfinite(trial[1]):
        axis.axvline(trial[0], color=TRIAL, linestyle=":", linewidth=1.1, label="Invalid trial value")
    if frame.direction:
        _draw_direction(axis, current, frame.direction, one_dimensional, xlim, ylim)
    style_legend(axis)
    return axis


def _plot_point(point, value, one_dimensional):
    pair = (point[0], value) if one_dimensional else point[:2]
    return tuple(float(x) if x is not None and isfinite(x) and abs(x) < PLOT_LIMIT else np.nan for x in pair)


def _plot_path(axis, steps, convergence, frame, current, following, trial, outline_trial) -> None:
    x, y = np.array(steps).T
    line, = axis.plot(x, y, color=PATH, linewidth=1.05, alpha=.58, marker="o", markersize=3,
                      label="iteration path", picker=5, zorder=4)
    line.set_gid("iterations")
    if convergence is not None:
        axis.scatter(*steps[convergence], marker="D", facecolors="none", edgecolors=CONVERGENCE,
                     s=130, linewidths=1.4, label="Convergence reached", zorder=5)
    axis.scatter(*current, s=58, facecolors=AXES, edgecolors=CURRENT, linewidths=1.5, zorder=7, label="xₖ")
    axis.annotate(f"{frame.current.iteration}", current, xytext=(-7, 8), textcoords="offset points",
                  color=CURRENT, ha="right", fontsize=8, zorder=9, annotation_clip=True)
    if following is not None:
        start, end = np.array(current), np.array(following)
        if np.all(np.isfinite([start, end])) and np.any(start != end):
            direction = end - start
            axis.quiver(*start, *direction, color=NEXT, angles="xy", scale_units="xy", scale=1,
                        width=.007, zorder=6)
        axis.scatter(*following, c=NEXT, edgecolors=AXES, s=55, label="xₖ₊₁", zorder=8)
        axis.annotate(f"{frame.following.iteration}", following, xytext=(7, -11), textcoords="offset points",
                      color=NEXT, fontsize=8, zorder=9, annotation_clip=True)
    if trial is not None and np.all(np.isfinite(trial)):
        if outline_trial:
            axis.scatter(*trial, facecolors="none", edgecolors=TRIAL, s=111, marker="s",
                         linewidths=1.6, label="Selected trial", zorder=9)
        elif trial not in (current, following):
            axis.scatter(*trial, c=TRIAL, edgecolors=AXES, s=55, marker="s", label="Selected trial", zorder=8)


def _draw_direction(axis, current, direction, one_dimensional, xlim, ylim) -> None:
    projected = np.array((direction[0], 0.) if one_dimensional else direction[:2])
    scale = np.max(np.abs(projected))
    if not scale or not np.all(np.isfinite(current)):
        return
    spans = (xlim[1] - xlim[0], ylim[1] - ylim[0])
    local_span = spans[0] if one_dimensional else min(spans)
    projected = projected / scale
    end = np.asarray(current) + .22 * local_span * projected / np.linalg.norm(projected)
    axis.annotate("", end, xytext=current,
                  arrowprops={"arrowstyle": "->", "color": DIRECTION, "linestyle": "--", "linewidth": 1.2},
                  zorder=6)
    axis.plot([], [], "--", color=DIRECTION, label="Direction (scaled)")


def plot_bounds(values, *, padding: float = .12) -> tuple[float, float]:
    finite = [float(x) for x in values if isfinite(x) and abs(x) < PLOT_LIMIT]
    if not finite:
        return -1., 1.
    low, high = min(finite), max(finite)
    scale = max(abs(low), abs(high), 1.)
    span = high - low
    margin = padding * max(span, 1e-10 * scale) if span else .2 * scale
    return low - margin, high + margin


def draw_contours(axis: Axes, mesh_x, mesh_y, values) -> None:
    finite = values[np.isfinite(values) & (np.abs(values) < PLOT_LIMIT)]
    if not finite.size:
        axis.text(.5, .5, "Function undefined on this view", transform=axis.transAxes,
                  color="#fca5a5", ha="center", va="center")
        return
    low, high = np.percentile(finite, (0, 97))
    if high == low:
        high = np.max(finite)
    levels = np.linspace(low, high, 23)
    masked = np.ma.masked_invalid(np.clip(values, low, high))
    if not np.all(np.diff(levels) > 0):
        axis.pcolormesh(mesh_x, mesh_y, masked, cmap="viridis", alpha=.35, shading="auto")
        return
    axis.contourf(mesh_x, mesh_y, masked, levels=levels, cmap="viridis", alpha=.68)
    axis.contour(mesh_x, mesh_y, masked, levels=levels[::4],
                 colors="#e2e8f0", linewidths=.4, alpha=.45)


def style_axis(axis: Axes, x_label: str, y_label: str) -> None:
    axis.set_facecolor(AXES)
    axis.set_xlabel(x_label, color="#cbd5e1", fontsize=9, labelpad=2)
    axis.set_ylabel(y_label, color="#cbd5e1", fontsize=9)
    axis.tick_params(colors="#cbd5e1", labelsize=8, pad=2)
    axis.xaxis.get_offset_text().set_color("#cbd5e1")
    axis.yaxis.get_offset_text().set_color("#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    axis.grid(color="#94a3b8", alpha=.17, linewidth=.6)


def style_legend(axis: Axes, *, columns: int = 2) -> None:
    if not axis.get_legend_handles_labels()[0]:
        return
    axis.legend(facecolor="#172033", edgecolor="#334155", labelcolor="#e2e8f0",
                fontsize=7, loc="best", ncol=columns, columnspacing=.9, handlelength=1.3)
