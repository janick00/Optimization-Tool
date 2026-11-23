"""Shared controls for step-by-step ILP examples."""

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QGridLayout, QGroupBox, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from topics.linear_programming.page import ProblemFormulationPage


class Playback(QWidget):
    selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.index = 0
        self.length = 0
        self.timer = QTimer(self)
        self.timer.setInterval(850)
        self.timer.timeout.connect(lambda: self.go(self.index + 1))
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.previous = QPushButton("Previous Step")
        self.next = QPushButton("Next Step")
        self.reset = QPushButton("Reset")
        self.auto = QPushButton("Play")
        for i, (button, action) in enumerate((
            (self.previous, lambda: self._manual_go(self.index - 1)),
            (self.next, lambda: self._manual_go(self.index + 1)),
            (self.reset, lambda: self._manual_go(0)), (self.auto, self._toggle),
        )):
            button.clicked.connect(action)
            layout.addWidget(button, i // 2, i % 2)
        self._sync()

    def start(self, length: int) -> None:
        self.stop()
        self.length = length
        self.index = 0
        self._sync()
        if length:
            self.selected.emit(0)

    def go(self, index: int) -> None:
        if 0 <= index < self.length:
            self.index = index
            if index == self.length - 1:
                self.stop()
            self._sync()
            self.selected.emit(index)

    def _manual_go(self, index: int) -> None:
        self.stop()
        self.go(index)

    def stop(self) -> None:
        self.timer.stop()
        self.auto.setText("Play")

    def _toggle(self) -> None:
        if self.timer.isActive():
            self.stop()
        elif self.index + 1 < self.length:
            self.timer.start()
            self.auto.setText("Pause")

    def _sync(self) -> None:
        self.previous.setEnabled(self.index > 0)
        self.next.setEnabled(self.index + 1 < self.length)
        self.reset.setEnabled(self.index > 0)
        self.auto.setEnabled(self.index + 1 < self.length)

    def hideEvent(self, event) -> None:
        self.stop()
        super().hideEvent(event)


def calculation_group() -> tuple[QGroupBox, QTextBrowser]:
    group = QGroupBox("Calculation Steps")
    group.setMinimumWidth(218)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(5, 6, 5, 5)
    view = QTextBrowser()
    view.setObjectName("stepsEditor")
    ProblemFormulationPage._set_document_font(view)
    layout.addWidget(view)
    return group, view


def lesson_columns(inputs: QWidget, visual: QWidget, calculations: QWidget) -> QSplitter:
    columns = QSplitter(Qt.Orientation.Horizontal)
    columns.setChildrenCollapsible(False)
    for column in (inputs, visual, calculations):
        columns.addWidget(column)
    columns.setSizes((300, 410, 280))
    return columns
