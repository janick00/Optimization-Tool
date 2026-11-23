"""Aitken fixed-point estimates in the desktop application."""

from __future__ import annotations

from math import isfinite

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QFormLayout, QGroupBox, QHeaderView, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QSplitter, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from topics.unconstrained_optimization.aitken_visualization import (
    plot_aitken_residuals,
    plot_aitken_values,
)
from topics.unconstrained_optimization.fixed_point_input import parse_fixed_point_function
from topics.unconstrained_optimization.methods.aitken import (
    DEFAULT_POINT_COUNT,
    MIN_POINT_COUNT,
    AitkenEstimate,
    FixedPointRun,
    iterate_fixed_point,
)


class AitkenPage(QWidget):
    """Inspect a scalar fixed-point sequence and its Aitken estimates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(12)
        title = QLabel("Aitken's Acceleration Method")
        title.setObjectName("topicTitle")
        layout.addWidget(title)
        layout.addWidget(self._create_setup())
        layout.addWidget(self._create_results(), stretch=1)
        self._run()

    def _create_setup(self) -> QGroupBox:
        group = QGroupBox("Fixed-point setup")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._function_edit = QLineEdit("cos(x)")
        self._function_edit.setToolTip("Use x as the variable, for example cos(x) or (x + 2) / 3.")
        form.addRow("Iteration function g(x)", self._function_edit)
        self._start_edit = QLineEdit("1")
        form.addRow("Initial value x₀", self._start_edit)
        self._points_spin = QSpinBox()
        self._points_spin.setRange(MIN_POINT_COUNT, 1000)
        self._points_spin.setValue(DEFAULT_POINT_COUNT)
        self._points_spin.setKeyboardTracking(False)
        self._points_spin.setAccessibleName("Number of Points")
        form.addRow("Number of Points", self._points_spin)

        actions = QHBoxLayout()
        run_button = QPushButton("Calculate")
        run_button.setObjectName("primaryButton")
        self._status_label = QLabel()
        self._status_label.setObjectName("optimizationStatus")
        self._status_label.setWordWrap(True)
        actions.addWidget(run_button)
        actions.addWidget(self._status_label, stretch=1)
        form.addRow(actions)
        run_button.clicked.connect(self._run)
        self._points_spin.valueChanged.connect(self._run)
        self._function_edit.returnPressed.connect(self._run)
        self._start_edit.returnPressed.connect(self._run)
        return group

    def _create_results(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        plot_group = QGroupBox("Fixed-point plots")
        plot_layout = QVBoxLayout(plot_group)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self._values_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self._residuals_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        for canvas, title in (
            (self._values_canvas, "Iteration values"), (self._residuals_canvas, "Residuals")
        ):
            canvas.setMinimumSize(240, 120)
            tabs.addTab(canvas, title)
        plot_layout.addWidget(tabs)
        splitter.addWidget(plot_group)

        table_group = QGroupBox("Aitken estimates")
        table_layout = QVBoxLayout(table_group)
        self._table = QTreeWidget()
        self._table.setHeaderLabels(["Iteration", "xⁱ⁻²", "xⁱ⁻¹", "xⁱ", "Aitken yⁱ", "Residual"])
        self._table.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.header().setMinimumSectionSize(78)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table.setWordWrap(False)
        self._table.setToolTip("Expand an iteration to see its denominator, substitution and residual.")
        self._table.setStyleSheet(
            "QTreeWidget { border: 1px solid #334155; border-radius: 5px; background: #0f172a; }"
            "QHeaderView::section { color: #93c5fd; background: #172033; padding: 5px; }"
        )
        table_layout.addWidget(self._table)
        splitter.addWidget(table_group)
        splitter.setSizes((360, 520))
        return splitter

    def _run(self) -> None:
        try:
            fixed_point = parse_fixed_point_function(self._function_edit.text())
            result = iterate_fixed_point(fixed_point, float(self._start_edit.text()), self._points_spin.value())
        except (TypeError, ValueError, ArithmeticError) as error:
            self._table.clear()
            for canvas in (self._values_canvas, self._residuals_canvas):
                canvas.figure.clear()
                canvas.figure.patch.set_facecolor("#111827")
                canvas.draw_idle()
            self._set_status(f"Input error: {error}", "error")
            return

        self._show_estimates(result)
        plot_aitken_values(self._values_canvas.figure, result)
        plot_aitken_residuals(self._residuals_canvas.figure, result)
        self._values_canvas.draw_idle()
        self._residuals_canvas.draw_idle()
        issues = sum(row.issue is not None for row in result.estimates)
        status = f"{len(result.points)} points · {len(result.estimates)} estimates"
        if issues:
            status += f" · {issues} with issues"
        reason = result.stop_reason or result.last_residual_issue
        if reason:
            status += f". {reason}"
        self._set_status(status, "warning" if issues or reason else "success")

    def _show_estimates(self, result: FixedPointRun) -> None:
        self._table.clear()
        for row in result.estimates:
            item = QTreeWidgetItem([
                str(row.index), *map(_number, row.values),
                _number(row.estimate) if row.estimate is not None else "Invalid",
                _number(row.residual),
            ])
            for column in range(1, self._table.columnCount()):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if row.issue:
                item.setToolTip(4, row.issue)
                item.setToolTip(5, row.issue)
            self._table.addTopLevelItem(item)
            details = QTreeWidgetItem(item, [_estimate_details(row)])
            details.setFirstColumnSpanned(True)

    def _set_status(self, status: str, state: str) -> None:
        self._status_label.setText(status)
        self._status_label.setProperty("state", state)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)


def _estimate_details(row: AitkenEstimate) -> str:
    older, previous, current = map(_number, row.values)
    denominator = _number(row.denominator)
    lines = [
        f"xⁱ⁻² = {older}   xⁱ⁻¹ = {previous}   xⁱ = {current}",
        f"D = xⁱ − 2xⁱ⁻¹ + xⁱ⁻² = {denominator}",
        "yⁱ = xⁱ − (xⁱ − xⁱ⁻¹)² / D",
    ]
    if row.estimate is not None:
        lines.append(f"yⁱ = {current} − ({current} − ({previous}))² / ({denominator})")
        lines.append(f"    = {_number(row.estimate)}")
    lines.append(f"Residual |g(yⁱ) − yⁱ| = {_number(row.residual)}")
    if row.issue:
        lines.append(row.issue)
    return "\n".join(lines)


def _number(value: float | None) -> str:
    if value is None or not isfinite(value):
        return "—"
    if 0 < abs(value) < 1e-6 or abs(value) >= 1e8:
        return f"{value:.8e}"
    return f"{value:.8f}"
