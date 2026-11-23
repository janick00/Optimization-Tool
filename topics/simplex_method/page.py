"""Simplex page reusing the structured linear-program input fields."""

from fractions import Fraction

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QScrollArea,
    QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from topics.linear_programming.models import ProblemForm
from topics.linear_programming.page import ProblemFormulationPage
from topics.linear_programming.problem_input import build_linear_program
from topics.simplex_method.formatting import format_calculations
from topics.simplex_method.solver import solve_simplex
from topics.simplex_method.visualization import plot_simplex, plot_simplex_message


class SimplexMethodPage(ProblemFormulationPage):
    """Solve a linear program from an automatically or manually selected vertex."""

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 28, 34, 28)
        outer.setSpacing(12)
        title = QLabel("Simplex Method")
        title.setObjectName("topicTitle")
        outer.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        inputs = QScrollArea()
        inputs.setObjectName("inputScrollArea")
        inputs.setFrameShape(QFrame.Shape.NoFrame)
        inputs.setWidgetResizable(True)
        inputs.setMinimumWidth(280)
        inputs.setWidget(self._create_input_group())
        self._convert_button.setText("Solve")
        splitter.addWidget(inputs)
        splitter.addWidget(self._create_output_group())
        splitter.setSizes((490, 430))
        outer.addWidget(splitter, stretch=1)

        self._variable_count_combo.currentIndexChanged.connect(self._handle_count_change)
        self._constraint_count_combo.currentIndexChanged.connect(self._handle_count_change)
        self._objective_sense_combo.currentIndexChanged.connect(self._convert)

    def _add_input_sections(self, layout: QVBoxLayout) -> None:
        self._start_edits: list[QLineEdit] = []
        self._start_values: dict[int, str] = {}
        title = QLabel("Starting Point")
        title.setObjectName("sectionLabel")
        layout.addWidget(title)

        self._start_mode_combo = QComboBox()
        self._start_mode_combo.addItem("Automatic", "automatic")
        self._start_mode_combo.addItem("Manual", "manual")
        self._start_mode_combo.setAccessibleName("Starting Point")
        layout.addWidget(self._start_mode_combo)

        self._start_fields = QWidget()
        self._start_layout = QHBoxLayout(self._start_fields)
        self._start_layout.setContentsMargins(0, 0, 0, 0)
        self._start_fields.hide()
        layout.addWidget(self._start_fields)
        self._start_mode_combo.currentIndexChanged.connect(self._change_start_mode)

    def _rebuild_dynamic_fields(self) -> None:
        super()._rebuild_dynamic_fields()
        self._start_values.update((i, edit.text()) for i, edit in enumerate(self._start_edits))
        self._clear_layout(self._start_layout)
        self._start_edits = []
        for i in range(len(self._objective_edits)):
            name = f"x{i + 1}"
            edit = self._create_number_edit(self._start_values.get(i, "0"))
            edit.setAccessibleName(f"Starting {name}")
            edit.setToolTip("Enter a decimal number or fraction.")
            edit.returnPressed.connect(self._convert)
            self._start_layout.addWidget(QLabel(f"{name} ="))
            self._start_layout.addWidget(edit)
            self._start_edits.append(edit)
        self._start_layout.addStretch()

    def _load_initial_example(self) -> None:
        self._load_default_inputs()

    def _update_form_fields(self) -> None:
        """The simplex input uses General Form with editable relations and domains."""

    def _change_start_mode(self) -> None:
        self._start_fields.setVisible(self._start_mode_combo.currentData() == "manual")
        self._convert()

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Simplex Calculations")
        layout = QVBoxLayout(group)
        self._status_label = QLabel()
        self._status_label.setObjectName("conversionStatus")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._canvas = FigureCanvas(Figure(figsize=(4, 3), facecolor="#111827"))
        self._canvas.setMinimumSize(225, 155)
        layout.addWidget(self._canvas, stretch=3)

        self._calculations = QTextBrowser()
        self._calculations.setObjectName("stepsEditor")
        self._set_document_font(self._calculations)
        layout.addWidget(self._calculations, stretch=2)
        return group

    def _manual_start(self) -> tuple[Fraction, ...] | None:
        if self._start_mode_combo.currentData() == "automatic":
            return None
        values = []
        for i, edit in enumerate(self._start_edits, 1):
            try:
                values.append(Fraction(edit.text().strip()))
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError(f"Starting Point x{i} must be a valid number or fraction.") from error
        return tuple(values)

    def _convert(self) -> None:
        try:
            objective, matrix, rhs, relations, domains = self._build_input_text()
            problem = build_linear_program(
                ProblemForm.GENERAL, self._objective_sense_combo.currentData(),
                objective, matrix, rhs, relations, domains,
            )
            if len(problem.objective) != len(self._objective_edits):
                raise ValueError("Enter a coefficient for every variable.")
            result = solve_simplex(problem, self._manual_start())
        except (ValueError, ArithmeticError) as error:
            message = f"Input or calculation error: {error}"
            self._calculations.clear()
            plot_simplex_message(self._canvas.figure, message)
            self._status_label.setText(message)
            self._status_label.setProperty("state", "error")
        else:
            self._calculations.setHtml(format_calculations(problem, result))
            plot_simplex(self._canvas.figure, problem, result)
            self._status_label.setText(result.message)
            self._status_label.setProperty(
                "state", "success" if result.status == "optimal" else "error"
            )
        self._refresh_style(self._status_label)
        self._canvas.draw_idle()
