"""Synchronized two-variable view of an LP or branch-and-bound frame."""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from topics.integer_programming.feasibility import PointCheck
from topics.integer_programming.formatting import point_text
from topics.integer_programming.hull_geometry import feasible_vertices, slice_interval
from topics.integer_programming.models import Domain, Node, TreeFrame
from topics.linear_programming.models import LinearProgram
from topics.simplex_method.solver import Inequality, SimplexResult, Vector, inequalities
from topics.simplex_method.visualization import _clip_polygon, plot_simplex_message


@dataclass(frozen=True)
class PlotRegion:
    rows: tuple[Inequality, ...]
    vertices: tuple[Vector, ...]


def prepare_region(problem: LinearProgram) -> PlotRegion:
    rows = inequalities(problem)
    vertices = feasible_vertices(rows) if len(problem.variable_names) == 2 else ()
    return PlotRegion(rows, vertices)


def plot_solution(
    figure: Figure, problem: LinearProgram, region: PlotRegion, domains: tuple[Domain, ...],
    *, frame: TreeFrame | None = None, selected: Node | None = None,
    simplex: SimplexResult | None = None, comparison: PointCheck | None = None,
) -> None:
    if len(problem.variable_names) != 2:
        plot_simplex_message(figure, "The feasible-region plot requires exactly two variables.")
        return
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _draw(figure, problem, region, domains, frame, selected, simplex, comparison)
    except (OverflowError, FloatingPointError):
        plot_simplex_message(figure, "The coordinates are outside the plot's numeric range.")


def _draw(figure, problem, region, domains, frame, selected, simplex, comparison):
    figure.clear()
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    markers = []
    if frame is not None:
        incumbent = frame.event.incumbent
        incumbent_label = "Optimal integer solution" if frame.event.outcome == "optimal" else "Incumbent"
        markers = [
            (frame.nodes[1].solution, "Root LP optimum", "#60a5fa", "o"),
            (incumbent.point if incumbent else None, incumbent_label, "#4ade80", "*"),
            (selected.solution if selected else None, "Selected LP solution", "#fb923c", "s"),
        ]
    elif simplex is not None and simplex.status == "optimal":
        markers = [(simplex.steps[-1].point, "LP optimum", "#60a5fa", "o")]
    if comparison is not None:
        label = "Manual point" if comparison.feasible else "Invalid manual point"
        color = "#c084fc" if comparison.feasible else "#f87171"
        markers.append((comparison.point, label, color, "D" if comparison.feasible else "X"))

    normalized = []
    anchors = [(0., 0.), *region.vertices, *(point for point, *_ in markers if point is not None)]
    if selected is not None:
        anchors.extend((branch.bound, 0) if branch.variable == 0 else (0, branch.bound)
                       for branch in selected.branches)
    for row in region.rows:
        scale = max(map(abs, row.coefficients))
        if not scale:
            normalized.append((np.zeros(2), float(row.bound)))
            continue
        normal = np.array([float(value / scale) for value in row.coefficients])
        bound = float(row.bound / scale)
        normalized.append((normal, bound))
        anchors.append(bound * normal / (normal @ normal))
    values = np.array(anchors, dtype=float)
    low, high = values.min(axis=0), values.max(axis=0)
    padding = np.maximum(high - low, 1) * .18
    low, high = low - padding, high + padding
    if np.max(np.abs([low, high])) >= 1e150:
        raise OverflowError("Plot range too large.")
    polygon = np.array([low, (high[0], low[1]), high, (low[0], high[1])])
    for normal, bound in normalized:
        polygon = _clip_polygon(polygon, normal, bound)
        if normal[0] == normal[1] == 0:
            continue
        if abs(normal[1]) >= abs(normal[0]):
            x = np.array([low[0], high[0]])
            axis.plot(x, (bound - normal[0] * x) / normal[1], "--", color="#64748b", linewidth=.8)
        else:
            y = np.array([low[1], high[1]])
            axis.plot((bound - normal[1] * y) / normal[0], y, "--", color="#64748b", linewidth=.8)
    if polygon.size:
        axis.fill(*polygon.T, color="#60a5fa", alpha=.18, label="LP feasible region", zorder=0)
    if frame is not None and frame.event.outcome == "optimal":
        _draw_discrete(axis, region.rows, domains, low, high)
    if selected is not None:
        for i, branch in enumerate(selected.branches):
            draw_line = axis.axvline if branch.variable == 0 else axis.axhline
            draw_line(branch.bound, color="#fb923c", linestyle=":", linewidth=1.6,
                      label="Selected branch limits" if i == 0 else None)
    for point, label, color, marker in markers:
        if point is not None:
            coordinates = tuple(map(float, point))
            axis.scatter(*coordinates, marker=marker, s={"o": 76, "*": 120, "s": 100, "D": 92, "X": 106}[marker],
                         edgecolors=color, facecolors="none", zorder=5, label=label)
            if frame is None:
                offset = (5, 7) if marker == "o" else (5, -13)
                axis.annotate(point_text(point), coordinates, xytext=offset, textcoords="offset points",
                              color=color, fontsize=8, zorder=6)
    axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    axis.set_xlabel(problem.variable_names[0], color="#cbd5e1")
    axis.set_ylabel(problem.variable_names[1], color="#cbd5e1")
    axis.tick_params(colors="#cbd5e1", labelsize=8)
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, facecolor="#172033", labelcolor="#f8fafc", fontsize=7)
    figure.tight_layout(pad=.7)


def _draw_discrete(axis, rows, domains, low, high):
    discrete = [i for i, domain in enumerate(domains) if domain is not Domain.CONTINUOUS]
    if not discrete:
        return
    fixed = discrete[0]
    other = 1 - fixed
    points, segments = [], []
    cap = 1500
    trimmed = False
    for k, value in enumerate(range(ceil(low[fixed]), floor(high[fixed]) + 1)):
        if k == cap:
            trimmed = True
            break
        interval = _slice(rows, fixed, value, Fraction(float(low[other])), Fraction(float(high[other])))
        if interval is None:
            continue
        lower, upper = interval
        if len(discrete) == 1:
            pair = ((value, float(lower)), (value, float(upper)))
            segments.append(pair if fixed == 0 else tuple(p[::-1] for p in pair))
        else:
            for coord in range(ceil(lower), floor(upper) + 1):
                if len(points) == cap:
                    trimmed = True
                    break
                points.append((value, coord) if fixed == 0 else (coord, value))
            if trimmed:
                break
    if points:
        axis.scatter(*zip(*points), s=11, color="#cbd5e1", alpha=.8, label="Feasible integer points")
    if segments:
        axis.add_collection(LineCollection(segments, colors="#cbd5e1", alpha=.45,
                                            linewidths=1, label="Feasible mixed-integer slices"))
    if trimmed:
        axis.text(.02, .02, "Discrete overlay limited to 1,500 entries", transform=axis.transAxes,
                  color="#cbd5e1", fontsize=7)


def _slice(rows, fixed, value, lower, upper):
    interval = slice_interval(rows, fixed, value)
    if interval is None:
        return None
    minimum, maximum = interval
    lower = max(lower, minimum) if minimum is not None else lower
    upper = min(upper, maximum) if maximum is not None else upper
    return (lower, upper) if lower <= upper else None
