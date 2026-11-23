"""Editable knapsack lesson with capacity playback and a selectable search tree."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from topics.integer_programming.knapsack import (
    LECTURE, Item, Knapsack, SearchOrder, density_order, greedy_relaxation, knapsack_branch_and_bound,
)
from topics.integer_programming.knapsack_presentation import (
    branch_label, greedy_frame, node_lines, plot_capacity, search_frame,
)
from topics.integer_programming.lesson_widgets import Playback, calculation_group, lesson_columns
from topics.integer_programming.tree_view import BranchTreeView


def _numeric(text: str) -> str:
    return text.replace(",", ".")


class KnapsackView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._problem = self._run = self._relaxed = None
        self._order = ()
        self._nodes = {}
        self._serial = 1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QHBoxLayout()
        self.status = QLabel()
        self.status.setObjectName("conversionStatus")
        self.status.setWordWrap(True)
        self.solve = QPushButton("Solve")
        self.solve.setObjectName("primaryButton")
        self.solve.clicked.connect(self._solve)
        heading.addWidget(self.status, 1)
        heading.addWidget(self.solve)
        layout.addLayout(heading)
        calculations, self.details = calculation_group()
        layout.addWidget(lesson_columns(self._input_group(), self._visual_group(), calculations), 1)
        self._load_lecture()

    def _input_group(self) -> QGroupBox:
        group = QGroupBox("Knapsack input")
        group.setMinimumWidth(218)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 6, 5, 5)
        self.method = QComboBox()
        self.method.setAccessibleName("Knapsack method")
        self.method.addItem("Greedy LP Relaxation", "lp")
        self.method.addItem("Branch and Bound", "bb")
        layout.addWidget(self.method)

        form = QGridLayout()
        form.setColumnStretch(1, 1)
        self.capacity = QLineEdit()
        self.capacity.setAccessibleName("Capacity")
        self.tolerance = QLineEdit("0")
        self.tolerance.setAccessibleName("Absolute objective tolerance")
        self.tolerance.setToolTip("Stop with a certified remaining objective gap at most this tolerance. Zero requests exact optimality.")
        self.limit = QSpinBox()
        self.limit.setAccessibleName("Maximum processed nodes")
        self.limit.setRange(1, 3000)
        self.limit.setValue(200)
        self.strategy = QComboBox()
        self.strategy.setAccessibleName("Branching strategy")
        self.strategy.setMinimumContentsLength(9)
        self.strategy.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for strategy in SearchOrder:
            self.strategy.addItem(strategy.value, strategy.value)
        for row, (name, field) in enumerate((
            ("Capacity b", self.capacity), ("Search", self.strategy),
            ("Tolerance", self.tolerance), ("Node limit", self.limit),
        )):
            form.addWidget(QLabel(name), row, 0)
            form.addWidget(field, row, 1)
        layout.addLayout(form)

        self.items = QTableWidget(0, 4)
        self.items.setMinimumHeight(105)
        self.items.setHorizontalHeaderLabels(("Item", "a", "c", "c/a"))
        for column, width in enumerate((51, 43, 43, 58)):
            self.items.setColumnWidth(column, width)
        self.items.verticalHeader().hide()
        self.items.setToolTip("The variable xⱼ is the item in row j. Edit its name, weight a or value c.")
        layout.addWidget(self.items, 1)
        buttons = QHBoxLayout()
        for text, action in (("Add Item", self._add_item), ("Remove", self._remove_item)):
            button = QPushButton(text)
            button.clicked.connect(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        load = QPushButton("Load Lecture Example")
        load.clicked.connect(self._load_lecture)
        layout.addWidget(load)
        self.method.currentIndexChanged.connect(self._invalidate)
        self.strategy.currentIndexChanged.connect(self._invalidate)
        self.capacity.textEdited.connect(self._invalidate)
        self.tolerance.textEdited.connect(self._invalidate)
        self.limit.valueChanged.connect(self._invalidate)
        self.items.itemChanged.connect(self._item_changed)
        return group

    def _visual_group(self) -> QGroupBox:
        group = QGroupBox("Knapsack Visualisation")
        group.setMinimumWidth(225)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 6, 5, 5)
        visual = QSplitter(Qt.Orientation.Vertical)
        visual.setChildrenCollapsible(False)
        self.tree = BranchTreeView(node_lines=node_lines, edge_label=branch_label, box_height=138)
        self.tree.setMinimumHeight(90)
        self.tree.node_selected.connect(self._inspect)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(3, 1.05), facecolor="#111827"))
        self.canvas.setMinimumSize(190, 92)
        visual.addWidget(self.tree)
        visual.addWidget(self.canvas)
        visual.setSizes((350, 125))
        visual.setStretchFactor(0, 4)
        visual.setStretchFactor(1, 1)
        layout.addWidget(visual, 1)
        self.tree_tools = QWidget()
        tools = QHBoxLayout(self.tree_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        for title, action in (("−", lambda: self.tree.zoom(.8)), ("+", lambda: self.tree.zoom(1.25)),
                              ("Fit Tree", self.tree.fit_tree)):
            button = QPushButton(title)
            button.clicked.connect(action)
            tools.addWidget(button)
        layout.addWidget(self.tree_tools)
        self.playback = Playback()
        self.playback.selected.connect(self._show_step)
        layout.addWidget(self.playback)
        return group

    def _write_item(self, row: int, name: str, weight: str, value: str) -> None:
        self.items.insertRow(row)
        for col, text in enumerate((name, weight, value, "")):
            cell = QTableWidgetItem(text)
            if col == 3:
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.items.setItem(row, col, cell)
        self._update_ratio(row)

    def _read_item(self, row: int) -> Item:
        name, weight, value = (self.items.item(row, col).text() for col in range(3))
        try:
            return Item(name, _numeric(weight), _numeric(value))
        except ValueError as error:
            raise ValueError(f"Item {row + 1}. {error}") from error

    def _update_ratio(self, row: int) -> None:
        try:
            ratio = str(self._read_item(row).density)
        except ValueError:
            ratio = "—"
        self.items.item(row, 3).setText(ratio)

    def _item_changed(self, cell: QTableWidgetItem) -> None:
        with QSignalBlocker(self.items):
            self._update_ratio(cell.row())
        self._invalidate()

    def _load_lecture(self) -> None:
        with QSignalBlocker(self.items):
            self.items.setRowCount(0)
            for row, item in enumerate(LECTURE.items):
                self._write_item(row, item.name, str(item.weight), str(item.value))
        self._serial = len(LECTURE.items) + 1
        self.capacity.setText(str(LECTURE.capacity))
        self._invalidate()

    def _add_item(self) -> None:
        row = self.items.rowCount()
        with QSignalBlocker(self.items):
            self._write_item(row, str(self._serial), "1", "1")
        self._serial += 1
        self.items.setCurrentCell(row, 0)
        self._invalidate()

    def _remove_item(self) -> None:
        if self.items.currentRow() >= 0:
            with QSignalBlocker(self.items):
                self.items.removeRow(self.items.currentRow())
            self._invalidate()

    def _invalidate(self) -> None:
        self._problem = self._run = self._relaxed = None
        self._nodes = {}
        self.playback.start(0)
        use_tree = self.method.currentData() == "bb"
        self.tree.setVisible(use_tree)
        self.tree_tools.setVisible(use_tree)
        self.tree.show_nodes({}, 0)
        for field in (self.strategy, self.tolerance, self.limit):
            field.setEnabled(use_tree)
        self.status.setText("Knapsack Problem · Ready")
        self.details.setPlainText("Edit the item weights and values, then select Solve.")
        self._draw(None)

    def _solve(self) -> None:
        self._invalidate()
        try:
            self._problem = Knapsack(tuple(self._read_item(row) for row in range(self.items.rowCount())),
                                     _numeric(self.capacity.text()))
            self._order = density_order(self._problem)
            if self.method.currentData() == "lp":
                self._relaxed = greedy_relaxation(self._problem, order=self._order)
                length = len(self._relaxed.prefix) + 4
            else:
                self._run = knapsack_branch_and_bound(
                    self._problem, SearchOrder(self.strategy.currentData()), node_limit=self.limit.value(),
                    tolerance=_numeric(self.tolerance.text()),
                )
                length = len(self._run.events)
        except (ValueError, ArithmeticError) as error:
            self.status.setText(f"Input error · {error}")
            return
        self.playback.start(length)

    def _show_step(self, index: int) -> None:
        if self._run:
            event = self._run.events[index]
            old_count = len(self._nodes)
            self._nodes = self._run.nodes_at(index)
            self.tree.show_nodes(self._nodes, event.focus)
            if index == 0 or (len(self._nodes) <= 9 and len(self._nodes) != old_count):
                self.tree.fit_tree()
            elif len(self._nodes) > 9:
                self.tree.center_on_node(event.focus)
            phase = event.outcome or event.phase.replace("_", " ")
            self.status.setText(f"Knapsack · Step {index} · {phase.capitalize()}")
            self._inspect(event.focus)
        else:
            details, point = greedy_frame(self._problem, self._order, self._relaxed, index)
            self.status.setText(f"Greedy LP Relaxation · Step {index} / {self.playback.length - 1}")
            self.details.setHtml(details)
            self._draw(point)

    def _inspect(self, node_id: int) -> None:
        if node_id not in self._nodes or self._run is None:
            return
        event = self._run.events[self.playback.index]
        node = self._nodes[node_id]
        point = node.relaxation.lp if node.relaxation else None
        if event.outcome == "optimal" and node_id == event.focus:
            point = event.incumbent
        self.details.setHtml(search_frame(self._problem, self._run, self.playback.index, self._nodes,
                                          node_id, self._order))
        self._draw(point)

    def _draw(self, point) -> None:
        plot_capacity(self.canvas.figure, self._problem, point, self._order)
        self.canvas.draw_idle()
