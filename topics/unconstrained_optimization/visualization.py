"""Matplotlib visualizations for unconstrained optimization runs."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from topics.unconstrained_optimization.solver import OptimizationResult, ParsedFunction


def plot_optimization_result(
    figure: Figure,
    function: ParsedFunction,
    result: OptimizationResult,
) -> None:
    """Draw the function and the optimization trajectory on a Matplotlib figure."""

    figure.clear()
    figure.patch.set_facecolor("#111827")
    points = np.asarray([step.point for step in result.steps], dtype=float)
    if function.dimension == 1:
        _plot_one_dimensional(figure, function, result, points)
    else:
        _plot_two_dimensional_slice(figure, function, result, points)
    figure.tight_layout(pad=1.2)


def _plot_one_dimensional(
    figure: Figure,
    function: ParsedFunction,
    result: OptimizationResult,
    points: np.ndarray,
) -> None:
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    values = points[:, 0]
    spread = max(float(np.ptp(values)), 1.0)
    lower = float(np.min(values) - 0.25 * spread - 0.5)
    upper = float(np.max(values) + 0.25 * spread + 0.5)
    grid = np.linspace(lower, upper, 500)
    function_values = np.asarray(
        [_safe_value(function, (value,)) for value in grid], dtype=float
    )
    axis.plot(grid, function_values, color="#60a5fa", linewidth=2.0, label="f(x)")
    point_values = np.asarray([step.value for step in result.steps], dtype=float)
    axis.plot(values, point_values, color="#f59e0b", linewidth=1.5, alpha=0.85)
    axis.scatter(values, point_values, color="#f97316", edgecolors="#fff7ed", zorder=4)
    for index, (x_value, y_value) in enumerate(zip(values, point_values)):
        axis.annotate(str(index), (x_value, y_value), color="#f8fafc", fontsize=8, xytext=(4, 4), textcoords="offset points")
    _style_axis(axis, function.variable_names[0], "f(x)")
    axis.set_title(f"{result.method.label} trajectory", color="#f8fafc")
    axis.legend(facecolor="#172033", edgecolor="#334155", labelcolor="#f8fafc")


def _plot_two_dimensional_slice(
    figure: Figure,
    function: ParsedFunction,
    result: OptimizationResult,
    points: np.ndarray,
) -> None:
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    x_values = points[:, 0]
    y_values = points[:, 1]
    x_spread = max(float(np.ptp(x_values)), 1.0)
    y_spread = max(float(np.ptp(y_values)), 1.0)
    x_lower = float(np.min(x_values) - 0.35 * x_spread - 0.5)
    x_upper = float(np.max(x_values) + 0.35 * x_spread + 0.5)
    y_lower = float(np.min(y_values) - 0.35 * y_spread - 0.5)
    y_upper = float(np.max(y_values) + 0.35 * y_spread + 0.5)
    x_grid = np.linspace(x_lower, x_upper, 110)
    y_grid = np.linspace(y_lower, y_upper, 110)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    fixed_values = points[0, 2:]
    surface_values = np.empty_like(mesh_x, dtype=float)
    for row in range(mesh_x.shape[0]):
        for column in range(mesh_x.shape[1]):
            point = (mesh_x[row, column], mesh_y[row, column], *fixed_values)
            surface_values[row, column] = _safe_value(function, point)

    finite_values = surface_values[np.isfinite(surface_values)]
    if finite_values.size:
        minimum = float(np.min(finite_values))
        upper_percentile = float(np.percentile(finite_values, 92))
        maximum = max(upper_percentile, minimum + 1e-9)
        levels = np.linspace(minimum, maximum, 24)
        axis.contourf(mesh_x, mesh_y, np.clip(surface_values, minimum, maximum), levels=levels, cmap="viridis", alpha=0.85)
        contours = axis.contour(mesh_x, mesh_y, np.clip(surface_values, minimum, maximum), levels=levels[::4], colors="#dbeafe", linewidths=0.35, alpha=0.5)
        axis.clabel(contours, inline=True, fontsize=7, fmt="%.2g", colors="#dbeafe")
    else:
        axis.text(0.5, 0.5, "Function could not be evaluated on the plot grid", transform=axis.transAxes, color="#fca5a5", ha="center", va="center")

    axis.plot(x_values, y_values, color="#f59e0b", linewidth=2.0, marker="o", markersize=4, label="iteration path")
    axis.scatter([x_values[0]], [y_values[0]], color="#22c55e", edgecolors="#f0fdf4", s=60, zorder=5, label="start")
    axis.scatter([x_values[-1]], [y_values[-1]], color="#ef4444", edgecolors="#fef2f2", s=60, zorder=5, label="result")
    if len(points) <= 25:
        for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
            axis.annotate(str(index), (x_value, y_value), color="#f8fafc", fontsize=8, xytext=(4, 4), textcoords="offset points")
    _style_axis(axis, function.variable_names[0], function.variable_names[1])
    fixed_text = ""
    if function.dimension > 2:
        fixed_text = " (2D slice; remaining variables fixed at x₀)"
    axis.set_title(f"{result.method.label} trajectory{fixed_text}", color="#f8fafc", fontsize=10)
    axis.legend(facecolor="#172033", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)


def _safe_value(function: ParsedFunction, point: tuple[float, ...]) -> float:
    try:
        return function.evaluate(point)
    except (ValueError, OverflowError, FloatingPointError):
        return float("nan")


def _style_axis(axis, x_label: str, y_label: str) -> None:
    axis.set_xlabel(x_label, color="#cbd5e1")
    axis.set_ylabel(y_label, color="#cbd5e1")
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    axis.grid(color="#475569", alpha=0.25, linewidth=0.6)
