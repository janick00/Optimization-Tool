"""Selectable record of the backtracking trials for the active iteration."""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QPlainTextEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from topics.unconstrained_optimization.formatting import _comparison_values, _compact_vector
from topics.unconstrained_optimization.history import IterationFrame


HEADERS = ("Trial", "β", "xₖ", "dₖ", "x_trial", "f(x_trial)", "f(xₖ)", "Improvement", "Status")


class BacktrackingTable(QWidget):
    trial_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._frame: IterationFrame | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        self.chain = QPlainTextEdit()
        self.chain.setReadOnly(True)
        self.chain.setAccessibleName("Tested beta sequence")
        self.chain.setStyleSheet("padding: 2px 7px;")
        self.chain.document().setDocumentMargin(1)
        self.chain.setFixedHeight(2 * self.chain.fontMetrics().lineSpacing() + 9)
        layout.addWidget(self.chain)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setAccessibleName("Successive halving backtracking trials")
        self.table.setToolTip("Select a row to inspect its candidate. Scroll horizontally for the comparison and status.")
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setMinimumHeight(72)
        for column, width in enumerate((47, 76, 93, 93, 118, 88, 77, 108, 86)):
            self.table.setColumnWidth(column, width)
        self.table.currentCellChanged.connect(self._current_cell_changed)
        self.table.cellClicked.connect(lambda row, _column: self.trial_selected.emit(row))
        layout.addWidget(self.table, stretch=1)

    def show_frame(self, frame: IterationFrame | None, selected: int | None) -> None:
        with QSignalBlocker(self.table):
            changed_frame = frame is not self._frame
            if changed_frame:
                self._frame = frame
                self._load_rows(frame)
            chosen = frame.chosen_trial if frame else None
            palette = self.table.palette()
            selection_color = "#246348" if chosen is not None and chosen == selected else "#294a6a"
            palette.setColor(QPalette.ColorRole.Highlight, QColor(selection_color))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#f8fafc"))
            self.table.setPalette(palette)
            if selected is not None and (changed_frame or self.table.currentRow() != selected):
                self.table.setCurrentCell(selected, max(0, self.table.currentColumn()))
            if changed_frame:
                self.table.verticalScrollBar().setValue(0)

    def _load_rows(self, frame: IterationFrame | None) -> None:
        trials = frame.trials if frame else ()
        self.table.clearContents()
        self.table.setRowCount(len(trials))
        if not trials:
            self.chain.setPlainText(f"Iteration {frame.number} · No trials at this point." if frame else "")
            return

        chain = []
        point = _compact_vector(frame.current.point, precision=8)
        direction = _compact_vector(frame.direction or (), precision=8)
        for row, trial in enumerate(trials):
            beta = repr(float(trial.step_size))
            chain.append(beta + (" (accepted)" if trial.accepted else ""))
            if trial.value is None:
                value, reference = "—", f"{frame.current.value:.10g}"
            else:
                value, reference = _comparison_values(trial.value, frame.current.value)
            improvement = trial.value is not None and trial.value < frame.current.value
            status = "Accepted" if trial.accepted else "Rejected"
            cells = (
                str(row + 1), beta, point, direction, _compact_vector(trial.point, precision=8),
                value, reference, "Yes" if improvement else "No", status,
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if trial.accepted:
                    item.setBackground(QBrush(QColor("#193b2a")))
                if column == len(HEADERS) - 1:
                    item.setForeground(QBrush(QColor("#86efac" if trial.accepted else "#fda4af")))
                    item.setToolTip(trial.rejection_reason or status)
                else:
                    item.setToolTip(text)
                self.table.setItem(row, column, item)
        self.chain.setPlainText(f"Iteration {frame.number} · β trials: " + " → ".join(chain))

    def _current_cell_changed(self, row: int, _column: int, previous_row: int, _previous_column: int) -> None:
        if row >= 0 and row != previous_row:
            self.trial_selected.emit(row)
