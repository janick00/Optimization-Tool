"""Qt controls and playback for the hull modes inside the ILP page."""

from dataclasses import replace
from math import isfinite

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from topics.integer_programming.cutting_planes import CuttingSession
from topics.integer_programming.hull_examples import ILP_EXAMPLES, POINT_EXAMPLES
from topics.integer_programming.hull_formatting import (
    format_construction, format_cutting_step, format_integer_hull,
)
from topics.integer_programming.hull_geometry import construct_hull
from topics.integer_programming.hull_plot import HullPlot, PlotOptions
from topics.integer_programming.integer_hull import compute_integer_hull
from topics.linear_programming.page import ProblemFormulationPage
from topics.linear_programming.problem_input import parse_matrix, parse_vector
from topics.simplex_method.visualization import plot_simplex_message


class HullControls(QWidget):
    mode_changed = Signal()
    example_requested = Signal(str)
    ranges_changed = Signal(object)
    manual_cut_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.method = QComboBox()
        self.method.setAccessibleName("Hull method")
        self.method.addItem("Convex Hull from Points", "points")
        self.method.addItem("Integer Hull", "integer")
        self.method.addItem("Cutting Planes", "cuts")
        layout.addWidget(self.method)

        example_row = QHBoxLayout()
        self.examples = QComboBox()
        self.examples.setAccessibleName("Hull examples")
        self.examples.setMinimumWidth(100)
        self.examples.currentTextChanged.connect(self.examples.setToolTip)
        load = QPushButton("Load")
        load.setAccessibleName("Load Example")
        load.setToolTip("Load the selected example.")
        load.clicked.connect(lambda: self.example_requested.emit(self.examples.currentText()))
        example_row.addWidget(self.examples, 1)
        example_row.addWidget(load)
        layout.addLayout(example_row)

        self.point_group = QWidget()
        points = QVBoxLayout(self.point_group)
        points.setContentsMargins(0, 0, 0, 0)
        self.points = QPlainTextEdit(POINT_EXAMPLES["Points inside a polygon"])
        self.points.setAccessibleName("2D points, one pair per line")
        self.points.setPlaceholderText("0, 0\n1, 2\n3, 1")
        self.points.setFixedHeight(115)
        points.addWidget(QLabel("Points (one x1, x2 pair per line)"))
        points.addWidget(self.points)
        layout.addWidget(self.point_group)

        self.limits_group = QWidget()
        limits = QGridLayout(self.limits_group)
        limits.setContentsMargins(0, 0, 0, 0)
        self.radius = self._spin(limits, 0, "Search radius ±R", 100, 10_000)
        self.radius.setToolTip("Limit for the absolute value of each discrete coordinate.")
        self.max_points = self._spin(limits, 1, "Maximum points", 1500, 50_000)
        self.max_points.setToolTip("Mixed-integer segments count as two endpoints.")
        layout.addWidget(self.limits_group)

        axes = QGridLayout()
        axes.setColumnStretch(1, 1)
        axes.setColumnStretch(2, 1)
        self.axis_edits = []
        for i, axis in enumerate(("x1", "x2")):
            axes.addWidget(QLabel(f"{axis} range"), i, 0)
            edits = []
            for j, bound in enumerate(("min", "max")):
                edit = QLineEdit()
                edit.setAccessibleName(f"{axis} {bound}")
                edit.setPlaceholderText(bound.capitalize())
                edit.setToolTip("Leave both bounds blank for an automatic axis range.")
                edit.setMinimumWidth(47)
                edit.editingFinished.connect(self._change_ranges)
                axes.addWidget(edit, i, j + 1)
                edits.append(edit)
            self.axis_edits.append(edits)
        layout.addLayout(axes)
        self.range_error = self._help(layout)

        self.cut_group = QWidget()
        cut_layout = QVBoxLayout(self.cut_group)
        cut_layout.setContentsMargins(0, 0, 0, 0)
        cut_layout.addWidget(QLabel("Manual cut: a₁x₁ + a₂x₂ ≤ b"))
        cut_row = QHBoxLayout()
        self.manual_cut = QLineEdit()
        self.manual_cut.setAccessibleName("Manual cut coefficients and bound")
        self.manual_cut.setPlaceholderText("a₁, a₂, b")
        self.add_cut = QPushButton("Check")
        self.add_cut.setAccessibleName("Check Cut")
        self.add_cut.clicked.connect(self._check_cut)
        self.manual_cut.returnPressed.connect(self._check_cut)
        cut_row.addWidget(self.manual_cut, 1)
        cut_row.addWidget(self.add_cut)
        cut_layout.addLayout(cut_row)
        self.cut_message = self._help(cut_layout)
        layout.addWidget(self.cut_group)
        self.method.currentIndexChanged.connect(self._update_mode)
        self._update_mode()

    @staticmethod
    def _spin(layout, row, name, default, maximum) -> QSpinBox:
        field = QSpinBox()
        field.setAccessibleName(name)
        field.setRange(1, maximum)
        field.setValue(default)
        layout.addWidget(QLabel(name), row, 0)
        layout.addWidget(field, row, 1)
        return field

    @staticmethod
    def _help(layout) -> QLabel:
        label = QLabel()
        label.setObjectName("fieldHelp")
        label.setWordWrap(True)
        label.hide()
        layout.addWidget(label)
        return label

    @property
    def mode(self) -> str:
        return self.method.currentData()

    def limits(self) -> dict[str, int]:
        return {"search_radius": self.radius.value(), "max_points": self.max_points.value()}

    def read_points(self):
        points = parse_matrix(self.points.toPlainText(), "Point list")
        if len(points[0]) != 2:
            raise ValueError("Each point requires exactly two coordinates.")
        return points

    def show_cut_access(self, enabled: bool) -> None:
        self.manual_cut.setEnabled(enabled)
        self.add_cut.setEnabled(enabled)
        self.cut_group.setToolTip("" if enabled else "First check integrality at a fractional LP optimum.")
        self.cut_message.hide()

    def show_cut_error(self, message: str) -> None:
        self.cut_message.setText(message)
        self.cut_message.show()

    def _update_mode(self) -> None:
        points = self.mode == "points"
        self.point_group.setVisible(points)
        self.limits_group.setVisible(not points)
        self.cut_group.setVisible(self.mode == "cuts")
        self.examples.clear()
        self.examples.addItems(POINT_EXAMPLES if points else ILP_EXAMPLES)
        self.mode_changed.emit()

    def _check_cut(self) -> None:
        if self.add_cut.isEnabled():
            self.manual_cut_requested.emit(self.manual_cut.text())

    def _change_ranges(self) -> None:
        try:
            ranges = []
            for i, (minimum, maximum) in enumerate(self.axis_edits, 1):
                values = (minimum.text().strip(), maximum.text().strip())
                if not any(values):
                    ranges.append(None)
                    continue
                parsed = parse_vector(", ".join(values), f"x{i} range")
                if len(parsed) != 2:
                    raise ValueError(f"Enter both bounds for x{i}, or leave both blank.")
                low, high = map(float, parsed)
                if not (-1e150 < low < high < 1e150 and isfinite(high - low)):
                    raise ValueError(f"The x{i} range requires finite bounds in increasing order.")
                ranges.append((low, high))
        except (ValueError, OverflowError) as error:
            self.range_error.setText(str(error))
            self.range_error.show()
            return
        self.range_error.hide()
        self.ranges_changed.emit(tuple(ranges))


class HullToolbar(NavigationToolbar2QT):
    toolitems = tuple(
        tool for tool in NavigationToolbar2QT.toolitems if tool[0] in ("Home", "Back", "Forward", "Pan", "Zoom")
    )

    def __init__(self, canvas, parent, reset_view):
        self._reset_view = reset_view
        super().__init__(canvas, parent, coordinates=False)

    def home(self, *_args) -> None:
        self._reset_view()


class HullWorkspace(QSplitter):
    status_changed = Signal(str)
    manual_access_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setChildrenCollapsible(False)
        self._construction = self._integer = self._cuts = self._plot = None
        self._problem = None
        self._index = 0
        self._options = PlotOptions()
        self._home_view = None

        plot_group = QGroupBox("Graphical Solution")
        center = QVBoxLayout(plot_group)
        self._canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3), facecolor="#111827"))
        self._canvas.setMinimumSize(200, 160)
        self._canvas.mpl_connect("motion_notify_event", self._hover)
        self._toolbar = HullToolbar(self._canvas, plot_group, self._home)
        center.addWidget(self._toolbar)
        center.addWidget(self._canvas, 1)
        switches = QGridLayout()
        self._lp_switch = self._switch(switches, 0, 0, "LP region", True)
        self._hull_switch = self._switch(switches, 0, 1, "Hull", True)
        self._coordinates_switch = self._switch(switches, 1, 0, "Coordinates", True)
        self._level_switch = self._switch(switches, 1, 1, "Level line", False)
        center.addLayout(switches)
        navigation = QGridLayout()
        self._previous = self._button(navigation, 0, 0, "Previous Step", self._back)
        self._next = self._button(navigation, 0, 1, "Next Step", self._forward)
        self._reset = self._button(navigation, 1, 0, "Reset", self._reset_steps)
        self._final = self._button(navigation, 1, 1, "Show Hull", self._show_hull)
        center.addLayout(navigation)
        self.addWidget(plot_group)

        calculation_group = QGroupBox("Calculation Steps")
        calculation_group.setMinimumWidth(220)
        right = QVBoxLayout(calculation_group)
        self.details = QTextBrowser()
        self.details.setObjectName("stepsEditor")
        ProblemFormulationPage._set_document_font(self.details)
        right.addWidget(self.details)
        self.addWidget(calculation_group)

    def _switch(self, layout, row, column, text, checked) -> QCheckBox:
        switch = QCheckBox(text)
        switch.setChecked(checked)
        switch.toggled.connect(self._draw)
        layout.addWidget(switch, row, column)
        return switch

    @staticmethod
    def _button(layout, row, column, name, action) -> QPushButton:
        button = QPushButton(name)
        button.clicked.connect(action)
        layout.addWidget(button, row, column)
        return button

    def clear(self, message: str) -> None:
        self._construction = self._integer = self._cuts = self._plot = None
        self._problem = self._home_view = None
        self._index = 0
        for button in (self._previous, self._next, self._reset, self._final):
            button.setEnabled(False)
        self.manual_access_changed.emit(False)
        self.details.setPlainText(message)
        self.status_changed.emit(message)
        plot_simplex_message(self._canvas.figure, "Enter valid input and press Solve.")
        self._canvas.draw_idle()

    def solve_points(self, points) -> None:
        self._construction = construct_hull(points)
        self._integer = self._cuts = self._problem = None
        self._index = 0
        self._update(fit=True)

    def solve_problem(self, mode, problem, domains, limits) -> None:
        self._construction = None
        self._problem = problem
        if mode == "integer":
            self._integer = compute_integer_hull(problem, domains, **limits)
            self._cuts = None
        else:
            self._cuts = CuttingSession(problem, domains, **limits)
            self._integer = None
        self._update(fit=True)

    def set_ranges(self, ranges) -> None:
        if self._options.ranges != ranges:
            self._options = replace(self._options, ranges=ranges)
            self._draw(fit=True)

    def add_cut(self, text: str) -> None:
        if self._cuts is None:
            return
        coefficients = parse_vector(text, "Manual cut")
        if len(coefficients) != 3:
            raise ValueError("Enter a₁, a₂, and the right-hand side b.")
        self._cuts.add_manual_cut(coefficients[:2], coefficients[2])
        self._update()

    def _back(self) -> None:
        if self._cuts:
            self._cuts.previous()
        else:
            self._index -= 1
        self._update()

    def _forward(self) -> None:
        if self._cuts:
            self._cuts.next()
        else:
            self._index += 1
        self._update()

    def _reset_steps(self) -> None:
        self._index = 0
        if self._cuts:
            self._cuts.reset()
        self._update(fit=True)

    def _show_hull(self) -> None:
        if self._construction:
            self._index = len(self._construction.steps) + 1
            self._update()

    def _update(self, *, fit=False) -> None:
        steps = self._construction is not None or self._cuts is not None
        can_manual = False
        previous = advance = False
        if self._construction:
            last = len(self._construction.steps) + 1
            details = format_construction(self._construction, self._index)
            status = f"Convex hull · Step {self._index} / {last}"
            previous, advance = self._index > 0, self._index < last
        elif self._cuts:
            details = format_cutting_step(self._cuts)
            frame = self._cuts.frame
            label = {
                "region": "Original LP relaxation", "solved": "LP solved", "checked": "Integrality checked",
                "preview": "Cut preview", "applied": "Cut added", "no_cut": "No automatic cut", "finished": "Finished",
            }[frame.phase]
            status = f"Cutting planes · Step {self._cuts.index} · {label}"
            previous, advance = self._cuts.index > 0, self._cuts.can_advance
            can_manual = self._cuts.can_add_manual
        else:
            assert self._integer and self._problem
            details = format_integer_hull(self._integer, self._problem.objective_sense)
            status = "Integer hull · " + {
                "complete": "Computed", "empty_lp": "Empty LP region", "empty_discrete": "Empty integer hull",
                "unbounded_region": "Unbounded LP region", "search_truncated": "Search range truncated",
                "point_limit": "Point limit reached",
            }[self._integer.status]
        self._lp_switch.setVisible(self._construction is None)
        self._level_switch.setVisible(self._construction is None)
        self._previous.setVisible(steps)
        self._next.setVisible(steps)
        self._reset.setVisible(steps)
        self._final.setVisible(self._construction is not None)
        self._previous.setEnabled(previous)
        self._next.setEnabled(advance)
        self._reset.setEnabled(previous)
        self._final.setEnabled(advance)
        self.manual_access_changed.emit(can_manual)
        self.details.setHtml(details)
        self.status_changed.emit(status)
        self._draw(fit=fit)

    def _draw(self, _checked=False, *, fit=False) -> None:
        if self._construction is None and self._integer is None and self._cuts is None:
            return
        figure = self._canvas.figure
        view = None if fit or not self._plot else (self._plot.axis.get_xlim(), self._plot.axis.get_ylim())
        if figure.axes:
            axis = figure.axes[0]
            axis.clear()
        else:
            axis = figure.add_subplot(111)
        axis.set_axis_on()
        figure.subplots_adjust(left=.19, right=.94, bottom=.26, top=.95)
        if fit:
            self._toolbar.update()
        self._options = replace(self._options, lp_region=self._lp_switch.isChecked(), hull=self._hull_switch.isChecked(),
                                coordinates=self._coordinates_switch.isChecked(), level_line=self._level_switch.isChecked())
        self._plot = HullPlot(axis, self._options)
        try:
            with np.errstate(over="raise", invalid="raise"):
                if self._construction:
                    self._plot.draw_construction(self._construction, self._index, view=view)
                elif self._cuts:
                    self._plot.draw_cutting(self._cuts, view=view)
                else:
                    self._plot.draw_integer(self._problem, self._integer, view=view)
                self._plot.finish()
            if fit or self._home_view is None:
                self._home_view = (axis.get_xlim(), axis.get_ylim())
        except (ArithmeticError, ValueError):
            self._plot = None
            plot_simplex_message(figure, "The coordinates are outside the plot's numeric range.")
        self._canvas.draw_idle()

    def _home(self) -> None:
        if self._plot and self._home_view:
            self._toolbar.push_current()
            self._plot.axis.set_xlim(*self._home_view[0])
            self._plot.axis.set_ylim(*self._home_view[1])
            self._toolbar.push_current()
            self._canvas.draw_idle()

    def _hover(self, event) -> None:
        if self._plot and self._plot.hover(event):
            self._canvas.draw_idle()
