"""The three formula-sheet steps for LP relaxation."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QLineEdit, QScrollArea, QTextBrowser, QToolButton,
    QVBoxLayout, QWidget,
)

from topics.integer_programming.feasibility import PointCheck
from topics.integer_programming.formatting import (
    format_comparison, format_relaxation, format_relaxation_range, format_simplex_trace,
)
from topics.linear_programming.models import LinearProgram, _as_fraction
from topics.simplex_method.solver import SimplexResult, Vector


class RelaxationSteps(QScrollArea):
    coordinates_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._names: tuple[str, ...] = ()
        self._coordinates: dict[str, str] = {}
        self._point_edits: list[QLineEdit] = []
        self._result: SimplexResult | None = None

        content = QWidget()
        self.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.addWidget(_heading("1. Optimierung mittels Simplex-Algorithmus lösen"))
        self._lp_summary = _summary()
        layout.addWidget(self._lp_summary)

        self._trace_toggle = QToolButton()
        self._trace_toggle.setText("Simplex-Zwischenrechnungen")
        self._trace_toggle.setCheckable(True)
        self._trace_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._trace_toggle.toggled.connect(self._update_trace)
        layout.addWidget(self._trace_toggle)
        self._trace = QTextBrowser()
        self._trace.setObjectName("stepsEditor")
        self._trace.setMinimumHeight(140)
        layout.addWidget(self._trace)
        self._update_trace(False)

        layout.addWidget(_heading("2. Eine zulässige ganzzahlige Lösung suchen"))
        layout.addWidget(_summary("Manuelle Eingabe · Koordinaten des Vergleichspunkts"))
        self._coordinates_layout = QGridLayout()
        layout.addLayout(self._coordinates_layout)
        self._point_summary = _summary()
        layout.addWidget(self._point_summary)

        layout.addWidget(_heading("3. Den Bereich des Optimums bestimmen"))
        self._bounds_summary = _summary()
        layout.addWidget(self._bounds_summary)
        layout.addStretch()

    def set_variables(self, names: tuple[str, ...]) -> None:
        if names == self._names:
            return
        self._coordinates.update((name, edit.text()) for name, edit in zip(self._names, self._point_edits))
        self._names = names
        while self._coordinates_layout.count():
            self._coordinates_layout.takeAt(0).widget().deleteLater()
        self._point_edits.clear()
        for i, name in enumerate(names):
            label = QLabel(name)
            edit = QLineEdit(self._coordinates.get(name, ""))
            edit.setAccessibleName(f"Vergleichspunkt {name}")
            edit.setToolTip("Dezimalzahl oder Bruch, zum Beispiel 3/2")
            label.setBuddy(edit)
            row, column = divmod(i, 2)
            self._coordinates_layout.addWidget(label, row, 2 * column)
            self._coordinates_layout.addWidget(edit, row, 2 * column + 1)
            self._point_edits.append(edit)
            edit.textChanged.connect(self.coordinates_changed)

    def read_point(self) -> tuple[Vector | None, str | None]:
        values = []
        missing = []
        for name, edit in zip(self._names, self._point_edits):
            text = edit.text().strip()
            if not text:
                missing.append(name)
                continue
            try:
                values.append(_as_fraction(text))
            except ValueError:
                return None, f"{name}: Gib eine gültige Zahl oder einen Bruch ein."
        if missing:
            return None, "Noch offen: " + ", ".join(missing) + "."
        return tuple(values), None

    def show_lp_result(self, problem: LinearProgram, result: SimplexResult) -> None:
        self._result = result
        self._lp_summary.setText(format_relaxation(problem, result))
        self._trace_toggle.setEnabled(bool(result.phases))
        self._update_trace(self._trace_toggle.isChecked())

    def show_comparison(
        self, problem: LinearProgram, result: SimplexResult, point: PointCheck | None, message: str | None,
    ) -> None:
        self._point_summary.setText(format_comparison(problem, point, message))
        self._bounds_summary.setText(format_relaxation_range(problem, result, point))

    def clear_result(self) -> None:
        self._result = None
        self._lp_summary.setText("Die LP-Relaxation ist noch nicht berechnet.")
        self._point_summary.setText("Für die Prüfung ist ein gültiges Optimierungsproblem erforderlich.")
        self._bounds_summary.setText("Derzeit keine Schranke verfügbar.")
        self._trace_toggle.setEnabled(False)
        self._update_trace(False)

    def _update_trace(self, expanded: bool) -> None:
        visible = expanded and self._result is not None and bool(self._result.phases)
        self._trace_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self._trace.setVisible(visible)
        if visible:
            self._trace.setHtml(format_simplex_trace(self._result))


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    label.setWordWrap(True)
    return label


def _summary(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
