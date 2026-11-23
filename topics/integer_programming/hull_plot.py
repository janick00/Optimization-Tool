"""Matplotlib drawing for convex hulls, integer hulls and cut playback."""

from dataclasses import dataclass
from itertools import chain
from math import isfinite

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import Polygon

from topics.integer_programming.cutting_planes import CuttingSession
from topics.integer_programming.formatting import point_text
from topics.integer_programming.hull_geometry import HullConstruction, dot
from topics.integer_programming.integer_hull import IntegerHull, Region
from topics.linear_programming.models import LinearProgram, ObjectiveSense
from topics.simplex_method.solver import Vector
from topics.simplex_method.visualization import _clip_polygon


BLUE, ORANGE, RED, TEAL = "#60a5fa", "#fb923c", "#f87171", "#2dd4bf"


@dataclass(frozen=True)
class PlotOptions:
    ranges: tuple[tuple[float, float] | None, tuple[float, float] | None] = (None, None)
    lp_region: bool = True
    hull: bool = True
    coordinates: bool = True
    level_line: bool = False


class HullPlot:
    def __init__(self, axis: Axes, options: PlotOptions) -> None:
        self.axis = axis
        self.options = options
        self.markers: list[tuple[PathCollection, tuple[Vector, ...]]] = []
        self.axis.set_facecolor("#0f172a")
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.tick_params(colors="#cbd5e1", labelsize=8)
        self.axis.grid(alpha=.2)
        self._hover = None
        self._hovered = None

    def finish(self) -> None:
        handles, labels = self.axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            short_labels = {
                "Feasible discrete points": "Discrete points",
                "Feasible mixed-integer segments": "Mixed-integer slices",
                "Current LP optimum": "LP optimum",
                "Maximize direction": "Max direction",
                "Minimize direction": "Min direction",
            }
            self.axis.legend(
                unique.values(), [short_labels.get(label, label) for label in unique],
                fontsize=7.5, facecolor="#172033", labelcolor="#f8fafc",
                loc="lower center", bbox_to_anchor=(.5, .02), bbox_transform=self.axis.figure.transFigure,
                ncol=2, columnspacing=.9, handlelength=1.4,
            )
        self._hover = self.axis.annotate(
            "", (0, 0), xytext=(8, 10), textcoords="offset points", color="#f8fafc", zorder=12,
            bbox={"boxstyle": "round,pad=.28", "facecolor": "#172033", "edgecolor": "#64748b"},
        )
        self._hover.set_visible(False)

    def hover(self, event) -> bool:
        if self._hover is None:
            return False
        hovered = None
        if event.inaxes is self.axis:
            for artist, points in reversed(self.markers):
                found, indices = artist.contains(event)
                if found:
                    hovered = points[indices["ind"][0]]
                    break
        if hovered == self._hovered:
            return False
        self._hovered = hovered
        self._hover.set_visible(hovered is not None)
        if hovered is not None:
            self._hover.xy = tuple(map(float, hovered))
            self._hover.set_text(point_text(hovered))
        return True

    def draw_construction(self, construction: HullConstruction, index: int, *, view=None) -> None:
        self._axes(construction.points, ("x1", "x2"), view=view)
        completed = index > len(construction.steps)
        vertices = construction.vertices if completed else ()
        if completed and self.options.hull:
            self._shape(vertices, ORANGE, "Convex hull", .25, zorder=2)
        points = tuple(dict.fromkeys(construction.points))
        if self.options.hull:
            selected = set(vertices)
            others = tuple(p for p in points if p not in selected)
            self._scatter(others, "#cbd5e1", "Other input points" if completed else "Input points",
                          size=25, annotate=len(points) <= 40)
            self._scatter(vertices, ORANGE, "Hull vertices", size=50, annotate=len(points) <= 40)
        else:
            self._scatter(points, "#cbd5e1", "Input points", size=25, annotate=len(points) <= 40)
        if 0 < index <= len(construction.steps):
            step = construction.steps[index - 1]
            lower, upper = construction.chains_at(index)
            for boundary, color, name in ((lower, TEAL, "Lower chain"), (upper, BLUE, "Upper chain")):
                if boundary:
                    self.axis.plot(*np.asarray(boundary, float).T, "o-", color=color, label=name,
                                   linewidth=1.5, markersize=4, zorder=4)
            if step.action == "remove":
                self._scatter((step.point,), RED, "Removed point", marker="x", size=85, annotate=True)
                assert step.candidate is not None
                self._scatter((step.candidate,), ORANGE, "Candidate", marker="D", size=65, annotate=True)
            else:
                self._scatter((step.point,), ORANGE, "Current point", marker="D", size=65, annotate=True)

    def draw_integer(self, problem: LinearProgram, result: IntegerHull, *, view=None) -> None:
        self._prepare_region(problem, result.region, view)
        if self.options.lp_region:
            self._region(result.region, BLUE, "LP region P", .26, zorder=0)
        self._discrete(result)
        lp = result.lp.optimum if result.lp else None
        optimum = result.discrete_optimum
        if lp is not None:
            self._scatter((lp.point,), BLUE, "LP optimum", size=90, annotate=True)
        if optimum is not None:
            coincides = lp is not None and optimum.point == lp.point
            name = "ILP / LP optimum" if coincides else "ILP optimum"
            self._scatter((optimum.point,), RED, name, marker="*", size=155, annotate=not coincides)
        self._objective(problem, lp)

    def draw_cutting(self, session: CuttingSession, *, view=None) -> None:
        frame = session.frame
        original = session.reference.region
        self._prepare_region(frame.problem, original, view)
        if self.options.lp_region:
            self._region(original, BLUE, "Original region P", .25, zorder=0)
        if frame.cuts:
            self._region(frame.region, TEAL, "Current LP region", .32, zorder=1)
        if frame.comparisons:
            self._discrete(session.reference)
        for cut in frame.cuts:
            self._line(cut.coefficients, cut.bound, TEAL, "--", "Applied cuts")
        if frame.candidate is not None:
            cut = frame.candidate
            self._line(cut.coefficients, cut.bound, RED, "-", "Proposed cut")
            rows = (*self._normal_rows(frame.region), self._normal(tuple(-c for c in cut.coefficients), -cut.bound))
            self._viewport_shape(rows, RED, "Removed area", .40, zorder=2)
        lp = frame.lp.optimum if frame.lp else None
        if lp:
            self._scatter((lp.point,), BLUE, "Current LP optimum", size=90, annotate=True)
        self._objective(frame.problem, lp)

    def _axes(self, points, names, *, view=None) -> None:
        locations = tuple(points) or ((0, 0),)
        column_limits = ((float(min(column)), float(max(column))) for column in zip(*locations))
        for axis_index, (low, high) in enumerate(column_limits):
            padding = max(high - low, abs(low) * 1e-13, 1.0) * .18
            limits = view[axis_index] if view is not None else self.options.ranges[axis_index]
            minimum, maximum = limits if limits is not None else (low - padding, high + padding)
            if not all(isfinite(v) and abs(v) < 1e150 for v in (minimum, maximum)):
                raise ValueError("The coordinates are outside the plot's numeric range.")
            (self.axis.set_xlim if axis_index == 0 else self.axis.set_ylim)(minimum, maximum)
        self.axis.set_xlabel(names[0], color="#cbd5e1")
        self.axis.set_ylabel(names[1], color="#cbd5e1")

    def _prepare_region(self, problem: LinearProgram, region: Region, view) -> None:
        anchors = list(region.vertices)
        if not region.bounded:
            anchors.append((0, 0))
            for row in region.rows:
                squared_norm = dot(row.coefficients, row.coefficients)
                if squared_norm:
                    anchors.append(tuple(c * row.bound / squared_norm for c in row.coefficients))
        self._axes(anchors, problem.variable_names, view=view)
        for row in region.rows:
            if any(row.coefficients):
                self._line(row.coefficients, row.bound, "#64748b", ":", "Constraints")

    def _discrete(self, result: IntegerHull) -> None:
        if not self.options.hull:
            return
        # With continuous-only inputs the hull coincides with P.
        name = "Integer hull" if result.points or result.segments else "Convex hull"
        self._shape(result.vertices, ORANGE, name, .32, zorder=2)
        self._scatter(result.points, "#020617", "Feasible discrete points", size=25)
        if result.segments:
            self.axis.add_collection(LineCollection(
                np.asarray(result.segments, float), colors="#fde68a", linewidths=1.4,
                label="Feasible mixed-integer segments", zorder=4,
            ))
            endpoints = tuple(chain.from_iterable(result.segments))
            self._scatter(endpoints, "#fde68a", None, size=18)
        if self.options.coordinates and len(result.vertices) <= 30:
            for point in result.vertices:
                self._label(point, ORANGE, offset=(-6, -12), align="right")

    def _region(self, region: Region, color: str, label: str, alpha: float, *, zorder: int) -> None:
        if region.bounded:
            self._shape(region.vertices, color, label, alpha, zorder=zorder)
        elif region.limits is not None:
            self._viewport_shape(self._normal_rows(region), color, label, alpha, zorder=zorder)

    def _shape(self, points, color, label, alpha, *, zorder: int) -> None:
        if len(points) >= 3:
            self.axis.add_patch(Polygon(np.asarray(points, float), facecolor=color, edgecolor=color,
                                        alpha=alpha, linewidth=2, label=label, zorder=zorder))
            self.axis.plot(*np.array((*points, points[0]), float).T, color=color, linewidth=1.4,
                           zorder=zorder + .1)
        elif len(points) == 2:
            self.axis.plot(*np.array(points, float).T, color=color, linewidth=2.2, label=label, zorder=zorder)
        elif points:
            self._scatter(points, color, label, size=55)

    @staticmethod
    def _normal(coefficients, bound):
        scale = max(map(abs, coefficients)) or 1
        return np.array([float(c / scale) for c in coefficients]), float(bound / scale)

    def _normal_rows(self, region):
        return tuple(self._normal(row.coefficients, row.bound) for row in region.rows)

    def _viewport_shape(self, rows, color, label, alpha, *, zorder: int) -> None:
        patch = Polygon(np.empty((0, 2)), facecolor=color, edgecolor=color,
                        alpha=alpha, linewidth=2, label=label, zorder=zorder)
        self.axis.add_patch(patch)

        def update(_axis):
            x0, x1 = self.axis.get_xlim()
            y0, y1 = self.axis.get_ylim()
            clipped = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            for normal, bound in rows:
                clipped = _clip_polygon(clipped, normal, bound)
            patch.set_xy(clipped.reshape(-1, 2))

        self.axis.callbacks.connect("xlim_changed", update)
        self.axis.callbacks.connect("ylim_changed", update)
        update(self.axis)

    def _line(self, coefficients, bound, color, style, label) -> None:
        # Pick a coordinate intercept along the larger normal component.
        axis = 0 if abs(coefficients[0]) >= abs(coefficients[1]) else 1
        normal, distance = self._normal(coefficients, bound)
        anchor = [0.0, 0.0]
        anchor[axis] = distance / normal[axis]
        if isfinite(anchor[axis]) and abs(anchor[axis]) < 1e150:
            self.axis.axline(anchor, slope=-normal[0] / normal[1] if normal[1] else np.inf,
                             color=color, linestyle=style, linewidth=1.35, label=label, zorder=3)

    def _scatter(self, points, color, label, *, size, marker="o", annotate=False) -> None:
        if not points:
            return
        scatter = self.axis.scatter(*np.asarray(points, float).T, color=color, marker=marker, s=size,
                                    linewidth=.8, edgecolors=None if marker == "x" else "#f1f5f9",
                                    label=label, zorder=5)
        self.markers.append((scatter, points))
        if annotate and self.options.coordinates:
            for point in points:
                self._label(point, color)

    def _label(self, point, color, offset=(5, 6), align="left") -> None:
        self.axis.annotate(point_text(point), tuple(map(float, point)), xytext=offset, ha=align,
                           textcoords="offset points", color=color, fontsize=7, zorder=6)

    def _objective(self, problem, optimum) -> None:
        coefficients = problem.objective
        if not any(coefficients):
            return
        normal, _ = self._normal(coefficients, 0)
        direction = normal / np.linalg.norm(normal)
        if problem.objective_sense is ObjectiveSense.MINIMIZE:
            direction = -direction
        color = "#bef264"
        if self.options.level_line and optimum:
            self._line(coefficients, optimum.value, color, "--", "Objective level line")
        arrow = self.axis.annotate(
            "", (0, 0), xytext=(0, 0), annotation_clip=False,
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.7}, zorder=6,
        )
        self.axis.plot([], [], color=color, marker=">", label=f"{problem.objective_sense.label} direction")

        def update(_axis):
            x0, x1 = self.axis.get_xlim()
            y0, y1 = self.axis.get_ylim()
            width, height = x1 - x0, y1 - y0
            start = np.array([x0 + width * .16, y0 + height * .16])
            arrow.set_position(tuple(start))
            arrow.xy = tuple(start + .12 * min(width, height) * direction)

        self.axis.callbacks.connect("xlim_changed", update)
        self.axis.callbacks.connect("ylim_changed", update)
        update(self.axis)
