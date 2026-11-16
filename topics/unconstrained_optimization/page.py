"""User interface for unconstrained continuous optimization."""

from __future__ import annotations

import re

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from topics.unconstrained_optimization.solver import (
    OptimizationMethod,
    OptimizationResult,
    OptimizationSettings,
    ParsedFunction,
    format_line_search_details_html,
    format_optimization_details_html,
    optimize,
    parse_function,
)
from topics.unconstrained_optimization.visualization import plot_optimization_result


class UnconstrainedOptimizationPage(QWidget):
    """Run a selected continuous-optimization method and inspect its history."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_function: ParsedFunction | None = None
        self._last_result: OptimizationResult | None = None
        self._build_ui()
        self._run_optimization()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(34, 28, 34, 28)
        outer_layout.setSpacing(12)

        title_label = QLabel("Unconstrained Optimization")
        title_label.setObjectName("topicTitle")
        subtitle_label = QLabel(
            "Explore gradient descent, Newton, and Broyden methods with analytic derivatives, "
            "plots, and iteration details."
        )
        subtitle_label.setObjectName("topicDescription")
        subtitle_label.setWordWrap(True)
        outer_layout.addWidget(title_label)
        outer_layout.addWidget(subtitle_label)

        outer_layout.addWidget(self._create_setup_group())
        outer_layout.addWidget(self._create_results_splitter(), stretch=1)

    def _create_setup_group(self) -> QGroupBox:
        group = QGroupBox("Optimization setup")
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setVerticalSpacing(8)

        self._method_combo = QComboBox()
        for method in OptimizationMethod:
            self._method_combo.addItem(method.label, method.value)
        layout.addRow("Method", self._method_combo)

        self._expression_edit = QLineEdit("(x1 - 2)**4 + (x1 - 2*x2)**2")
        self._expression_edit.setPlaceholderText("e.g. (x1 - 2)**4 + (x1 - 2*x2)**2")
        self._expression_edit.setToolTip(
            "Enter a SymPy-compatible formula, for example x1**2 + sin(x2)."
        )
        layout.addRow("Function f(x)", self._expression_edit)

        variable_start_layout = QHBoxLayout()
        self._variables_edit = QLineEdit("x1, x2")
        self._variables_edit.setPlaceholderText("x1, x2")
        variable_start_layout.addWidget(self._variables_edit, stretch=1)
        variable_start_layout.addWidget(QLabel("Start point x0"))
        self._start_point_edit = QLineEdit("0, 0")
        self._start_point_edit.setPlaceholderText("0, 0")
        variable_start_layout.addWidget(self._start_point_edit, stretch=1)
        layout.addRow("Variables", variable_start_layout)

        parameter_layout = QHBoxLayout()
        parameter_layout.addWidget(QLabel("Startwert β₀"))
        self._initial_step_edit = QLineEdit("1.0")
        self._initial_step_edit.setFixedWidth(90)
        self._initial_step_edit.setPlaceholderText("z. B. 0.25")
        self._initial_step_edit.setToolTip(
            "Choose the first beta value yourself. Each later trial uses β/2 or 2β."
        )
        parameter_layout.addWidget(self._initial_step_edit)
        parameter_layout.addWidget(QLabel("Tolerance"))
        self._tolerance_edit = QLineEdit("1e-6")
        self._tolerance_edit.setFixedWidth(90)
        parameter_layout.addWidget(self._tolerance_edit)
        parameter_layout.addWidget(QLabel("Max. iterations"))
        self._max_iterations_spin = QSpinBox()
        self._max_iterations_spin.setRange(1, 1000)
        self._max_iterations_spin.setValue(30)
        self._max_iterations_spin.setFixedWidth(90)
        parameter_layout.addWidget(self._max_iterations_spin)
        parameter_layout.addStretch()
        layout.addRow("Parameters", parameter_layout)

        action_layout = QHBoxLayout()
        self._run_button = QPushButton("Run optimization")
        self._run_button.setObjectName("primaryButton")
        action_layout.addWidget(self._run_button)
        self._status_label = QLabel()
        self._status_label.setObjectName("optimizationStatus")
        self._status_label.setWordWrap(True)
        action_layout.addWidget(self._status_label, stretch=1)
        layout.addRow("", action_layout)

        self._run_button.clicked.connect(self._run_optimization)
        return group

    def _create_results_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        plot_group = QGroupBox("Function and optimization trajectory")
        plot_layout = QVBoxLayout(plot_group)
        self._figure = Figure(figsize=(5, 4), facecolor="#111827")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumSize(420, 300)
        plot_layout.addWidget(self._canvas)
        splitter.addWidget(plot_group)

        result_group = QGroupBox("Intermediate results")
        result_layout = QVBoxLayout(result_group)
        self._intermediate_browser = QTextBrowser()
        self._intermediate_browser.setObjectName("optimizationDetails")
        self._intermediate_browser.setReadOnly(True)
        intermediate_font = QFont("DejaVu Serif")
        intermediate_font.setStyleHint(QFont.StyleHint.Serif)
        intermediate_font.setPointSize(9)
        self._intermediate_browser.setFont(intermediate_font)
        result_layout.addWidget(self._intermediate_browser, stretch=3)

        details_label = QLabel("Formulas, derivatives and optimization summary")
        details_label.setObjectName("sectionLabel")
        result_layout.addWidget(details_label)
        self._details_browser = QTextBrowser()
        self._details_browser.setObjectName("optimizationDetails")
        self._details_browser.setReadOnly(True)
        details_font = QFont("DejaVu Serif")
        details_font.setStyleHint(QFont.StyleHint.Serif)
        details_font.setPointSize(9)
        self._details_browser.setFont(details_font)
        result_layout.addWidget(self._details_browser, stretch=2)
        splitter.addWidget(result_group)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes((520, 500))
        return splitter

    def _run_optimization(self) -> None:
        try:
            function = parse_function(
                self._expression_edit.text(),
                self._variables_edit.text(),
            )
            start = self._parse_point(self._start_point_edit.text())
            settings = OptimizationSettings(
                initial_step=float(self._initial_step_edit.text()),
                tolerance=float(self._tolerance_edit.text()),
                max_iterations=self._max_iterations_spin.value(),
            )
            method = OptimizationMethod(self._method_combo.currentData())
            result = optimize(function, method, start, settings)
        except (ValueError, TypeError, OverflowError) as error:
            self._show_error(str(error))
            return

        self._last_function = function
        self._last_result = result
        self._intermediate_browser.setHtml(format_line_search_details_html(result))
        self._details_browser.setHtml(format_optimization_details_html(result))
        plot_optimization_result(self._figure, function, result)
        self._canvas.draw_idle()
        self._status_label.setText(
            f"{result.message} Final point: {self._format_point(result.final_point)}"
        )
        self._status_label.setProperty("state", "success" if result.converged else "warning")
        self._refresh_style(self._status_label)

    def _show_error(self, message: str) -> None:
        self._status_label.setText(f"Input or calculation error: {message}")
        self._status_label.setProperty("state", "error")
        self._intermediate_browser.clear()
        self._details_browser.clear()
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        axis.set_facecolor("#0f172a")
        axis.text(0.5, 0.5, "Run a valid optimization to display the plot", color="#94a3b8", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        self._canvas.draw_idle()
        self._refresh_style(self._status_label)

    @staticmethod
    def _parse_point(text: str) -> tuple[float, ...]:
        tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
        if not tokens:
            raise ValueError("Enter a starting point, for example 0, 0.")
        return tuple(float(token) for token in tokens)

    @staticmethod
    def _format_point(point: tuple[float, ...]) -> str:
        return "(" + ", ".join(f"{value:.6g}" for value in point) + ")"

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
