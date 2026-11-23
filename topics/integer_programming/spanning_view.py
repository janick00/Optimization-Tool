"""Editable undirected graph and stepwise comparison of connector formulations."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from topics.integer_programming.lesson_widgets import Playback, calculation_group, lesson_columns
from topics.integer_programming.spanning_formulations import (
    LECTURE_GRAPH, LECTURE_PARTITION, TRIANGLE, build_connector_models, choose_partition,
    polytope_three, solve_connector_lp, verify_integer_optimum,
)
from topics.integer_programming.spanning_presentation import (
    STEPS, ConnectorGraphView, format_spanning, plot_polytopes,
)
from topics.network_optimization.models import Graph, parse_graph


class SpanningView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._graph = self._models = self._weak = self._strong = self._optimum = None
        self._polytopes = {}
        self._rows = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName("conversionStatus")
        self.solve = QPushButton("Solve")
        self.solve.setObjectName("primaryButton")
        self.solve.clicked.connect(self._solve)
        heading.addWidget(self.status, 1)
        heading.addWidget(self.solve)
        layout.addLayout(heading)
        calculations, self.details = calculation_group()
        layout.addWidget(lesson_columns(self._input_group(), self._visual_group(), calculations), 1)
        self._load_example()

    def _input_group(self) -> QGroupBox:
        group = QGroupBox("Graph input")
        group.setMinimumWidth(215)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 6, 5, 5)
        examples = QHBoxLayout()
        self.examples = QComboBox()
        self.examples.setAccessibleName("Spanning tree example")
        self.examples.setMinimumContentsLength(8)
        self.examples.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.examples.addItem("Lecture Triangle (pp. 24, 26)", "triangle")
        self.examples.addItem("Lecture Graph (pp. 22, 25)", "lecture")
        self.examples.currentTextChanged.connect(self.examples.setToolTip)
        load = QPushButton("Load")
        load.setAccessibleName("Load spanning tree example")
        load.clicked.connect(self._load_example)
        examples.addWidget(self.examples, 1)
        examples.addWidget(load)
        layout.addLayout(examples)

        count_row = QHBoxLayout()
        self.count = QSpinBox()
        self.count.setAccessibleName("Number of nodes")
        self.count.setRange(1, 20)
        self.count.setKeyboardTracking(False)
        count_row.addWidget(QLabel("Nodes"))
        count_row.addWidget(self.count, 1)
        layout.addLayout(count_row)
        self.nodes = QTableWidget(0, 1)
        self.nodes.setHorizontalHeaderLabels(("Name",))
        self.nodes.horizontalHeader().setStretchLastSection(True)
        self.nodes.verticalHeader().setDefaultSectionSize(24)
        self.nodes.setMinimumHeight(76)
        layout.addWidget(self.nodes, 1)
        remove_node = QPushButton("Remove Node")
        remove_node.clicked.connect(self._remove_node)
        layout.addWidget(remove_node)

        edge_label = QLabel("Edges by node number")
        edge_label.setToolTip("E.g. 1, 2 selects the names in rows 1 and 2. Each undirected edge appears once.")
        layout.addWidget(edge_label)
        self.edges = QTableWidget(0, 3)
        self.edges.setHorizontalHeaderLabels(("From", "To", "Weight"))
        self.edges.verticalHeader().hide()
        self.edges.setMinimumHeight(95)
        for i, width in enumerate((49, 47, 69)):
            self.edges.setColumnWidth(i, width)
        self.edges.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.edges, 2)
        actions = QHBoxLayout()
        for text, action in (("Add Edge", self._add_edge), ("Remove Edge", self._remove_edge)):
            button = QPushButton(text)
            button.clicked.connect(action)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.count.valueChanged.connect(self._resize_nodes)
        self.nodes.itemChanged.connect(self._refresh_graph)
        self.edges.itemChanged.connect(self._refresh_graph)
        return group

    def _visual_group(self) -> QGroupBox:
        group = QGroupBox("Graphical Solution")
        group.setMinimumWidth(224)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 6, 5, 5)
        self.graph = ConnectorGraphView()
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3), facecolor="#111827"))
        self.views = QStackedWidget()
        self.views.addWidget(self.graph)
        self.views.addWidget(self.canvas)
        layout.addWidget(self.views, 1)
        tools = QHBoxLayout()
        self.graph_tools = QWidget()
        graph_tools = QHBoxLayout(self.graph_tools)
        graph_tools.setContentsMargins(0, 0, 0, 0)
        for title, action in (("−", lambda: self.graph.zoom(.8)), ("+", lambda: self.graph.zoom(1.25)),
                              ("Fit", self.graph.fit_graph)):
            button = QPushButton(title)
            button.clicked.connect(action)
            graph_tools.addWidget(button)
        tools.addWidget(self.graph_tools, 1)
        self.diagram = QCheckBox("3D")
        self.diagram.setAccessibleName("Show 3D polytopes")
        self.diagram.setToolTip("Show the 3D polytopes. Drag to rotate.")
        self.diagram.toggled.connect(self._show_plot)
        tools.addWidget(self.diagram)
        layout.addLayout(tools)
        self.rows = QComboBox()
        self.rows.setAccessibleName("Illustrated cut or partition row")
        self.rows.currentIndexChanged.connect(self._render)
        layout.addWidget(self.rows)
        self.playback = Playback()
        self.playback.selected.connect(self._step)
        layout.addWidget(self.playback)
        return group

    def _load_example(self) -> None:
        lecture = self.examples.currentData() == "lecture"
        size, rows = (9, LECTURE_GRAPH) if lecture else (3, TRIANGLE)
        with QSignalBlocker(self.count), QSignalBlocker(self.nodes), QSignalBlocker(self.edges):
            self.count.setValue(size)
            self.nodes.setRowCount(size)
            self.edges.setRowCount(0)
            for i in range(size):
                self.nodes.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            for u, v, weight in rows:
                self._append_edge((u + 1, v + 1, weight))
        self._refresh_graph(reset_positions=True)
        if lecture:
            self.details.setPlainText("Nine-node graph from the lecture with example positive weights.")

    def _append_edge(self, cells=("", "", "")) -> None:
        row = self.edges.rowCount()
        with QSignalBlocker(self.edges):
            self.edges.insertRow(row)
            for column, value in enumerate(cells):
                self.edges.setItem(row, column, QTableWidgetItem(str(value)))

    def _add_edge(self) -> None:
        self._append_edge()
        self.edges.setCurrentCell(self.edges.rowCount() - 1, 0)
        self.edges.editItem(self.edges.currentItem())

    def _remove_edge(self) -> None:
        if self.edges.currentRow() >= 0:
            self.edges.removeRow(self.edges.currentRow())
            self._refresh_graph()

    def _resize_nodes(self, size: int) -> None:
        old = self.nodes.rowCount()
        with QSignalBlocker(self.nodes):
            self.nodes.setRowCount(size)
            names = {self.nodes.item(i, 0).text() for i in range(min(size, old))}
            for i in range(old, size):
                name = str(i + 1)
                while name in names:
                    name += "'"
                self.nodes.setItem(i, 0, QTableWidgetItem(name))
                names.add(name)
        if size < old:
            self._update_endpoints(size)
        self._refresh_graph()

    def _remove_node(self) -> None:
        index = self.nodes.currentRow()
        if index < 0 or self.nodes.rowCount() == 1:
            return
        with QSignalBlocker(self.nodes), QSignalBlocker(self.count):
            self.nodes.removeRow(index)
            self.count.setValue(self.nodes.rowCount())
        self._update_endpoints(self.nodes.rowCount(), index + 1)
        self._refresh_graph()

    def _update_endpoints(self, size: int, removed: int | None = None) -> None:
        with QSignalBlocker(self.edges):
            for row in reversed(range(self.edges.rowCount())):
                cells = tuple(self.edges.item(row, col) for col in (0, 1))
                try:
                    endpoints = tuple(int(cell.text()) for cell in cells)
                except ValueError:
                    continue
                if (removed is None and max(endpoints) > size) or removed in endpoints:
                    self.edges.removeRow(row)
                elif removed is not None:
                    for cell, value in zip(cells, endpoints):
                        if value > removed:
                            cell.setText(str(value - 1))

    def _read_graph(self) -> Graph:
        size = self.nodes.rowCount()
        names = tuple(self.nodes.item(i, 0).text() for i in range(size))
        matrix = [[""] * size for _ in names]
        for row in range(self.edges.rowCount()):
            a, b, weight = (self.edges.item(row, col).text().strip() for col in range(3))
            if not (a or b or weight):
                continue
            try:
                u, v = int(a) - 1, int(b) - 1
            except ValueError as error:
                raise ValueError(f"Edge row {row + 1} requires two node numbers.") from error
            if not (0 <= u < size and 0 <= v < size and u != v):
                raise ValueError(f"Edge row {row + 1} requires two different nodes from 1 to {size}.")
            if matrix[u][v]:
                raise ValueError(f"Edge row {row + 1} duplicates another undirected edge.")
            if not weight or weight.lower() in ("inf", "+inf", "∞", "+∞"):
                raise ValueError(f"Edge row {row + 1} requires a finite weight.")
            matrix[u][v] = matrix[v][u] = weight
        return parse_graph(names, tuple(map(tuple, matrix)), False)

    def _refresh_graph(self, _item=None, *, reset_positions=False) -> None:
        self._models = self._weak = self._strong = self._optimum = None
        self._polytopes.clear()
        self.playback.start(0)
        self.rows.hide()
        self.diagram.setVisible(False)
        self.diagram.setChecked(False)
        self.views.setCurrentWidget(self.graph)
        try:
            self._graph = self._read_graph()
        except ValueError as error:
            self._graph = None
            self.graph.set_graph(None)
            self.status.setText(f"Input error · {error}")
            self.details.setPlainText(str(error))
            return
        self.graph.set_graph(self._graph, reset_positions=reset_positions)
        self.status.setText("Spanning Tree / ILP Formulations · Ready")
        self.details.setPlainText("Edit the graph, then select Solve to study its formulations.")

    def _solve(self) -> None:
        self._refresh_graph()
        if self._graph is None:
            self.details.setPlainText("Correct the graph input to begin.")
            return
        illustration = LECTURE_PARTITION if self._graph.names == tuple(map(str, range(1, 10))) else None
        self._models = build_connector_models(self._graph, illustration=illustration)
        self.playback.start(len(STEPS))

    def _polytope(self, strong: bool):
        if strong not in self._polytopes:
            self._polytopes[strong] = polytope_three(self._models, strong=strong)
        return self._polytopes[strong]

    def _witness(self):
        point = self._weak.point
        if point is not None and any(x.denominator != 1 for x in point):
            return point
        if len(self._models.edges) == 3:
            return next((p for p in self._polytope(False).vertices
                         if any(x.denominator != 1 for x in p)), None)
        return None

    def _step(self, index: int) -> None:
        models = self._models
        diagram_available = index >= 4 and len(models.edges) == 3 and models.connected
        self.diagram.setVisible(diagram_available)
        if not diagram_available or index == len(STEPS) - 1:
            with QSignalBlocker(self.diagram):
                self.diagram.setChecked(False)
        if index >= 3 and self._weak is None:
            self._weak = solve_connector_lp(models)
        if index >= 6 and self._strong is None:
            self._strong = solve_connector_lp(models, strong=True, weak_bound=self._weak.bound)
        if index == 8 and self._optimum is None:
            self._optimum = verify_integer_optimum(models)
        self._rows = models.cuts if index == 2 else models.additions if index == 5 else ()
        with QSignalBlocker(self.rows):
            self.rows.clear()
            for i, row in enumerate(self._rows, 1):
                self.rows.addItem(f"Row {i} · {len(row.groups)} groups")
            if index == 5 and self._rows:
                selected = choose_partition(models, self._weak)
                self.rows.setCurrentIndex(self._rows.index(selected))
        self.rows.setVisible(bool(self._rows))
        self._render()
        self._show_plot()

    def _render(self, _row=0) -> None:
        if self._models is None:
            return
        index = self.playback.index
        row = self._rows[self.rows.currentIndex()] if self._rows and self.rows.currentIndex() >= 0 else None
        witness = self._witness() if 4 <= index <= 5 else None
        weak_point = self._weak.point if self._weak is not None else None
        point = (witness if witness is not None else weak_point) if 4 <= index <= 5 else None
        if 6 <= index <= 7 and self._strong is not None:
            point = self._strong.point
        optimum = self._optimum if index == 8 else None
        edges = optimum.tree if optimum else None
        self.graph.show_formulation(self._graph, point=point, row=row, selected=edges)
        self.status.setText(f"Spanning Tree · Step {index} / {len(STEPS) - 1} · {STEPS[index]}")
        self.details.setHtml(format_spanning(self._models, index, self._weak, self._strong, row, witness, optimum,
                                             diagram=not self.diagram.isHidden()))

    def _show_plot(self, _checked=False) -> None:
        show = self.diagram.isChecked() and not self.diagram.isHidden()
        self.views.setCurrentWidget(self.canvas if show else self.graph)
        self.graph_tools.setEnabled(not show)
        if show:
            regions = [("V1 · Cuts", self._polytope(False))]
            if self.playback.index >= 6:
                regions.append(("V2 · Partitions", self._polytope(True)))
            plot_polytopes(self.canvas.figure, regions)
            self.canvas.draw_idle()
