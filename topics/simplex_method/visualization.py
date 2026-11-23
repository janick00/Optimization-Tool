"""Plot a two-variable feasible region and the simplex path."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from topics.linear_programming.models import LinearProgram
from topics.simplex_method.solver import SimplexResult, inequalities


def plot_simplex(figure: Figure, problem: LinearProgram, result: SimplexResult) -> None:
    if len(problem.variable_names) != 2:
        plot_simplex_message(figure, "The feasible-region plot is available for two variables.")
        return
    try:
        _plot_region(figure, problem, result)
    except (OverflowError, FloatingPointError):
        plot_simplex_message(figure, "The coordinates are outside the plot's numeric range.")


def plot_simplex_message(figure: Figure, message: str) -> None:
    figure.clear()
    axis = figure.add_subplot(111)
    axis.set_axis_off()
    axis.text(.5, .5, message, color="#cbd5e1", ha="center", va="center",
              wrap=True, transform=axis.transAxes)


def _plot_region(figure: Figure, problem: LinearProgram, result: SimplexResult) -> None:
    figure.clear()
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    rows = inequalities(problem)
    bad_rows = {i for i, check in enumerate(result.start.checks) if not check.satisfied} if result.start else set()
    geometry = []
    anchors = [np.array([0.0, 0.0])]
    if result.start is not None:
        anchors.append(np.array(result.start.point, dtype=float))
    path = np.array([step.point for step in result.steps], dtype=float)
    if path.size:
        anchors.extend(path)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index, row in enumerate(rows):
            scale = max(map(abs, row.coefficients))
            if not scale:
                geometry.append((np.zeros(2), float(row.bound), index in bad_rows))
                continue
            normal = np.array([float(value / scale) for value in row.coefficients])
            bound = float(row.bound / scale)
            anchor = (bound / np.dot(normal, normal)) * normal
            anchors.append(anchor)
            geometry.append((normal, bound, index in bad_rows))

        lower = np.min(anchors, axis=0)
        upper = np.max(anchors, axis=0)
        pad = np.maximum(upper - lower, 1.0) * .15 + .25
        lower, upper = lower - pad, upper + pad
        if np.max(np.abs([lower, upper])) >= 1e150:
            raise OverflowError("Plot range too large.")
        polygon = np.array([
            lower, (upper[0], lower[1]), upper, (lower[0], upper[1]),
        ])
        for normal, bound, violated in geometry:
            polygon = _clip_polygon(polygon, normal, bound)
            if not np.any(normal):
                continue
            color = "#f87171" if violated else "#64748b"
            if abs(normal[1]) >= abs(normal[0]):
                x = np.array([lower[0], upper[0]])
                axis.plot(x, (bound - normal[0] * x) / normal[1], "--", color=color, linewidth=.8)
            else:
                y = np.array([lower[1], upper[1]])
                axis.plot((bound - normal[1] * y) / normal[0], y, "--", color=color, linewidth=.8)

    if polygon.size:
        axis.fill(*polygon.T, color="#60a5fa", alpha=.2, label="Feasible region in view")
    if path.size:
        axis.plot(*path.T, "-o", color="#f59e0b", linewidth=2, markersize=4, label="Simplex path")
    if result.start is not None:
        start = result.start
        point = np.array(start.point, dtype=float)
        color = "#f87171" if start.error else "#4ade80"
        label = "Invalid start" if start.error else "Starting point"
        axis.scatter(*point, color=color, edgecolors="#f8fafc", zorder=5, s=48, label=label)
        coords = ", ".join(map(str, start.point))
        axis.annotate(f"({coords})", point, xytext=(5, 7), textcoords="offset points",
                      fontsize=8, color="#f8fafc")
        if start.error:
            axis.text(.02, .98, start.error, va="top", color="#fecaca", fontsize=8,
                      wrap=True, transform=axis.transAxes)
    else:
        axis.text(.02, .98, result.message, va="top", color="#fecaca", fontsize=8,
                  wrap=True, transform=axis.transAxes)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_xlabel(problem.variable_names[0], color="#cbd5e1")
    axis.set_ylabel(problem.variable_names[1], color="#cbd5e1")
    axis.tick_params(colors="#cbd5e1", labelsize=8)
    axis.grid(alpha=.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, facecolor="#172033", labelcolor="#f8fafc", fontsize=7)
    figure.tight_layout(pad=.8)


def _clip_polygon(polygon: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray:
    """Clip a convex polygon to one bounded half-plane in plot coordinates."""

    if not polygon.size:
        return polygon
    clipped = []
    previous = polygon[-1]
    before = float(previous @ normal - bound)
    for point in polygon:
        after = float(point @ normal - bound)
        if (before <= 0) != (after <= 0):
            clipped.append(previous + (before / (before - after)) * (point - previous))
        if after <= 0:
            clipped.append(point)
        previous, before = point, after
    return np.array(clipped)
