"""Matrix editing, graph preview and reversible algorithm playback."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from topics.network_optimization.formatting import format_step
from topics.network_optimization.models import Graph, Method, Step, is_missing, parse_graph
from topics.network_optimization.solver import run_algorithm
from topics.network_optimization.visualization import NetworkView


EXAMPLE_NAMES = ("A", "B", "C", "D", "E")
EXAMPLE_EDGES = ((0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8),
                 (2, 4, 10), (3, 4, 2))


class NetworkOptimizationPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._graph: Graph | None = None
        self._steps: tuple[Step, ...] = ()
        self._step_index = 0
        self._build_ui()
        self._load_example()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 18, 26, 18)
        layout.setSpacing(8)
        title = QLabel("Graphen & Netzwerke")
        title.setObjectName("topicTitle")
        layout.addWidget(title)
        layout.addLayout(self._build_controls())
        self._status = QLabel()
        self._status.setObjectName("conversionStatus")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.setChildrenCollapsible(False)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        top.addWidget(self._build_matrix())
        top.addWidget(self._build_graph())
        top.setSizes((420, 475))
        vertical.addWidget(top)
        vertical.addWidget(self._build_results())
        vertical.setSizes((340, 225))
        layout.addWidget(vertical, stretch=1)

    def _build_controls(self) -> QGridLayout:
        controls = QGridLayout()
        self._method_combo = QComboBox()
        for method in Method:
            self._method_combo.addItem(method.value, method.value)
        self._method_combo.currentIndexChanged.connect(self._method_changed)
        controls.addWidget(QLabel("Verfahren"), 0, 0)
        controls.addWidget(self._method_combo, 0, 1, 1, 3)
        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 20)
        self._count_spin.setKeyboardTracking(False)
        self._count_spin.setAccessibleName("Anzahl Knoten")
        self._count_spin.valueChanged.connect(self._resize_matrix)
        controls.addWidget(QLabel("Anzahl Knoten"), 0, 4)
        controls.addWidget(self._count_spin, 0, 5)

        self._start_combo = QComboBox()
        self._target_combo = QComboBox()
        for column, label, combo in ((0, "Start", self._start_combo), (2, "Ziel", self._target_combo)):
            combo.setMinimumContentsLength(6)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.currentIndexChanged.connect(self._invalidate_playback)
            controls.addWidget(QLabel(label), 1, column)
            controls.addWidget(combo, 1, column + 1)
        self._directed_check = QCheckBox("Gerichtet")
        self._directed_check.toggled.connect(self._direction_changed)
        controls.addWidget(self._directed_check, 1, 4, 1, 2)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        return controls

    def _build_matrix(self) -> QGroupBox:
        group = QGroupBox("Adjazenzmatrix")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        self._matrix = QTableWidget()
        self._matrix.setMinimumSize(220, 100)
        self._matrix.horizontalHeader().setDefaultSectionSize(70)
        self._matrix.verticalHeader().setDefaultSectionSize(28)
        self._matrix.setWordWrap(False)
        self._matrix.setToolTip("Erste Spalte: Knotennamen. Leer oder ∞: keine Kante. 0: Kante mit Gewicht 0.")
        self._matrix.itemChanged.connect(self._cell_changed)
        layout.addWidget(self._matrix, stretch=1)
        buttons = QHBoxLayout()
        for text, action in (("Beispiel laden", self._load_example), ("Matrix leeren", self._clear_matrix)):
            button = QPushButton(text)
            button.clicked.connect(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return group

    def _build_graph(self) -> QGroupBox:
        group = QGroupBox("Graph")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        self._network_view = NetworkView()
        layout.addWidget(self._network_view, stretch=1)
        tools = QHBoxLayout()
        for text, action in (("−", lambda: self._network_view.zoom(.8)),
                             ("+", lambda: self._network_view.zoom(1.25)),
                             ("Einpassen", self._network_view.fit_graph)):
            button = QPushButton(text)
            button.clicked.connect(action)
            tools.addWidget(button)
        tools.addStretch()
        layout.addLayout(tools)
        return group

    def _build_results(self) -> QGroupBox:
        group = QGroupBox("Berechnungsschritte")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        buttons = QHBoxLayout()
        self._run_button = QPushButton("Berechnen")
        self._run_button.setObjectName("primaryButton")
        self._first = QPushButton("Anfang")
        self._previous = QPushButton("Vorheriger Schritt")
        self._next = QPushButton("Nächster Schritt")
        self._last = QPushButton("Endergebnis")
        for button, action in (
            (self._run_button, self._calculate), (self._first, lambda: self._show_step(0)),
            (self._previous, lambda: self._show_step(self._step_index - 1)),
            (self._next, lambda: self._show_step(self._step_index + 1)),
            (self._last, lambda: self._show_step(len(self._steps) - 1)),
        ):
            button.clicked.connect(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._results = QTextBrowser()
        self._results.setObjectName("stepsEditor")
        self._results.setMinimumHeight(62)
        layout.addWidget(self._results, stretch=1)
        return group

    def _load_example(self) -> None:
        size = len(EXAMPLE_NAMES)
        weights = [[""] * size for _ in range(size)]
        for u, v, weight in EXAMPLE_EDGES:
            weights[u][v] = weights[v][u] = str(weight)
        with QSignalBlocker(self._count_spin), QSignalBlocker(self._directed_check):
            self._count_spin.setValue(size)
            self._directed_check.setChecked(False)
        self._write_matrix(EXAMPLE_NAMES, weights, reset_positions=True)

    def _clear_matrix(self) -> None:
        names, _ = self._read_matrix()
        self._write_matrix(names, [[""] * len(names) for _ in names])

    def _resize_matrix(self, size: int) -> None:
        names, previous = self._read_matrix()
        next_names = list(names[:size])
        for i in range(len(next_names), size):
            base = chr(ord("A") + i)
            name = base
            suffix = 2
            while name in {value.strip() for value in next_names}:
                name = f"{base}{suffix}"
                suffix += 1
            next_names.append(name)
        weights = [[previous[i][j] if i < len(names) and j < len(names) else ""
                    for j in range(size)] for i in range(size)]
        self._write_matrix(next_names, weights)

    def _write_matrix(self, names, weights, *, reset_positions=False) -> None:
        size = len(names)
        with QSignalBlocker(self._matrix):
            self._matrix.setRowCount(size)
            self._matrix.setColumnCount(size + 1)
            self._matrix.setHorizontalHeaderLabels(["Knoten", *names])
            for i, name in enumerate(names):
                self._matrix.setItem(i, 0, QTableWidgetItem(name))
                for j in range(size):
                    item = QTableWidgetItem("0" if i == j else weights[i][j])
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if i == j:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._matrix.setItem(i, j + 1, item)
        self._refresh_graph(reset_positions=reset_positions)

    def _read_matrix(self):
        size = self._matrix.rowCount()
        names = tuple(self._matrix.item(i, 0).text() for i in range(size))
        values = tuple(tuple(self._matrix.item(i, j + 1).text() for j in range(size)) for i in range(size))
        return names, values

    def _cell_changed(self, item: QTableWidgetItem) -> None:
        i, j = item.row(), item.column() - 1
        if j >= 0 and i != j and not self._directed_check.isChecked():
            with QSignalBlocker(self._matrix):
                self._matrix.item(j, i + 1).setText(item.text())
        self._refresh_graph()

    def _direction_changed(self, directed: bool) -> None:
        conflict = False
        if not directed:
            with QSignalBlocker(self._matrix):
                for i in range(self._matrix.rowCount()):
                    for j in range(i + 1, self._matrix.rowCount()):
                        upper, lower = self._matrix.item(i, j + 1), self._matrix.item(j, i + 1)
                        a, b = upper.text(), lower.text()
                        conflict |= not is_missing(a) and not is_missing(b) and a.strip() != b.strip()
                        common = b if is_missing(a) and not is_missing(b) else a
                        upper.setText(common)
                        lower.setText(common)
        self._refresh_graph()
        if conflict and self._graph is not None:
            self._set_status("Bei widersprüchlichen Kanten gilt die obere Matrixhälfte.", "")

    def _refresh_graph(self, *, reset_positions: bool = False) -> None:
        names, weights = self._read_matrix()
        self._matrix.setHorizontalHeaderLabels(["Knoten", *(name.strip() for name in names)])
        self._set_endpoint_options(names)
        try:
            self._graph = parse_graph(names, weights, self._directed_check.isChecked())
        except ValueError as error:
            self._graph = None
            self._invalidate_playback()
            self._network_view.set_graph(None)
            self._set_status(str(error), "error")
            return
        self._network_view.set_graph(self._graph, reset_positions=reset_positions)
        self._method_changed()

    def _set_endpoint_options(self, names) -> None:
        for combo, optional in ((self._start_combo, False), (self._target_combo, True)):
            current = combo.currentData()
            with QSignalBlocker(combo):
                combo.clear()
                if optional:
                    combo.addItem("Kein Ziel", None)
                for i, name in enumerate(names):
                    combo.addItem(name.strip() or f"Knoten {i + 1}", i)
                index = combo.findData(current)
                combo.setCurrentIndex(index if index >= 0 else 0)

    def _method(self) -> Method:
        return Method(self._method_combo.currentData())

    def _method_changed(self) -> None:
        method = self._method()
        self._start_combo.setEnabled(method.uses_start)
        self._target_combo.setEnabled(method.uses_target)
        self._directed_check.setEnabled(not method.requires_undirected or self._directed_check.isChecked())
        self._invalidate_playback()

    def _invalidate_playback(self) -> None:
        self._steps = ()
        self._step_index = 0
        self._network_view.set_step(None)
        self._update_buttons()
        self._results.setPlainText("Berechnen startet das gewählte Verfahren. Leer oder ∞ steht für eine fehlende Kante.")
        if self._graph is not None:
            issue = self._configuration_issue()
            self._set_status(issue or f"{self._graph.size} Knoten · {len(self._graph.edges())} Kanten",
                             "error" if issue else "")

    def _configuration_issue(self) -> str | None:
        if self._graph is None:
            return "Korrigiere zuerst die Adjazenzmatrix."
        if self._method().requires_undirected and self._graph.directed:
            return "Dieses Verfahren benötigt einen ungerichteten Graphen."
        if self._method() is Method.DIJKSTRA and any(self._graph.weight(e) < 0 for e in self._graph.edges()):
            return "Dijkstra benötigt nichtnegative Kantengewichte."
        return None

    def _calculate(self) -> None:
        if self._graph is None:
            return
        try:
            self._steps = run_algorithm(
                self._graph, self._method(), self._start_combo.currentData(),
                self._target_combo.currentData() if self._method().uses_target else None,
            )
        except ValueError as error:
            self._set_status(str(error), "error")
            return
        self._show_step(0)

    def _show_step(self, index: int) -> None:
        if self._graph is None or not 0 <= index < len(self._steps):
            return
        self._step_index = index
        step = self._steps[index]
        self._results.setHtml(format_step(self._graph, step, index, len(self._steps)))
        self._network_view.set_step(step)
        self._set_status(f"Schritt {index + 1}/{len(self._steps)} · {step.title}", "")
        self._update_buttons()

    def _update_buttons(self) -> None:
        self._run_button.setEnabled(self._configuration_issue() is None)
        for button in (self._first, self._previous):
            button.setEnabled(bool(self._steps) and self._step_index > 0)
        for button in (self._next, self._last):
            button.setEnabled(self._step_index + 1 < len(self._steps))

    def _set_status(self, text: str, state: str) -> None:
        self._status.setText(text)
        self._status.setProperty("state", state)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)
