"""ILP editor, chronological tree controls and synchronized node inspection."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from topics.integer_programming.feasibility import check_integer_point
from topics.integer_programming.formatting import format_node, format_simplex_trace
from topics.integer_programming.hull_examples import ILP_EXAMPLES, POINT_EXAMPLES
from topics.integer_programming.hull_view import HullControls, HullWorkspace
from topics.integer_programming.knapsack_view import KnapsackView
from topics.integer_programming.models import Domain, TreeFrame, TreeRun
from topics.integer_programming.relaxation_view import RelaxationSteps
from topics.integer_programming.solver import DEFAULT_NODE_LIMIT, branch_and_bound, relaxation
from topics.integer_programming.spanning_view import SpanningView
from topics.integer_programming.tree_view import COLORS, BranchTreeView
from topics.integer_programming.visualization import plot_solution, prepare_region
from topics.linear_programming.models import ProblemForm
from topics.linear_programming.page import ProblemFormulationPage
from topics.linear_programming.problem_input import build_linear_program, parse_vector
from topics.simplex_method.solver import SimplexResult, solve_simplex
from topics.simplex_method.visualization import plot_simplex_message


class IntegerProgrammingPage(ProblemFormulationPage):
    """Host ILP methods and the two dedicated combinatorial optimization lessons."""

    _DOMAIN_OPTIONS = tuple((domain.value, domain.value) for domain in Domain)
    _LESSONS = {"knapsack": KnapsackView, "spanning": SpanningView}

    def _load_initial_example(self) -> None:
        self._load_default_inputs()
        self._general_domain_values = [Domain.INTEGER.value] * 2

    def _update_form_fields(self) -> None:
        """Relations and integrality remain editable in this editor."""

    def _create_domain_combo(self, current_value: str) -> QComboBox:
        combo = super()._create_domain_combo(current_value)
        combo.setFixedWidth(116)
        combo.setToolTip("All variables are nonnegative; Binary also imposes x ≤ 1.")
        return combo

    def _build_ui(self) -> None:
        self._run: TreeRun | None = None
        self._frame: TreeFrame | None = None
        self._lp_result: SimplexResult | None = None
        self._step = 0
        self._selected = 1
        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 20, 26, 20)
        title = QLabel("Integer Linear Programming (ILP)")
        title.setObjectName("topicTitle")
        heading = QHBoxLayout()
        heading.addWidget(title, 1)
        self._mode_combo = QComboBox()
        self._mode_combo.setAccessibleName("ILP method")
        self._mode_combo.addItem("LP Relaxation", "lp")
        self._mode_combo.addItem("Branch and Bound", "bb")
        self._mode_combo.addItem("Convex Hull / Cutting Planes", "hull")
        self._mode_combo.addItem("Knapsack Problem", "knapsack")
        self._mode_combo.addItem("Spanning Tree / ILP Formulations", "spanning")
        self._mode_combo.setCurrentIndex(1)
        heading.addWidget(self._mode_combo)
        outer.addLayout(heading)
        self._status_label = QLabel()
        self._status_label.setObjectName("conversionStatus")
        self._status_label.setWordWrap(True)
        self._status_row = QHBoxLayout()
        self._status_row.addWidget(self._status_label, 1)
        outer.addLayout(self._status_row)

        vertical = QSplitter(Qt.Orientation.Vertical)
        self._vertical_splitter = vertical
        vertical.setChildrenCollapsible(False)
        top = QSplitter(Qt.Orientation.Horizontal)
        self._standard_top = top
        top.setChildrenCollapsible(False)
        inputs = QScrollArea()
        self._input_scroll = inputs
        inputs.setObjectName("inputScrollArea")
        inputs.setFrameShape(QFrame.Shape.NoFrame)
        inputs.setWidgetResizable(True)
        inputs.setMinimumSize(215, 115)
        input_group = self._create_input_group()
        input_group.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._editor_fields.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        inputs.setWidget(input_group)
        self._convert_button.setText("Solve")
        top.addWidget(inputs)
        top.addWidget(self._create_output_group())
        top.setSizes((420, 420))
        vertical.addWidget(top)

        self._lower = QSplitter(Qt.Orientation.Horizontal)
        self._lower.setChildrenCollapsible(False)
        self._tree_group = self._create_tree_group()
        self._lower.addWidget(self._tree_group)
        graph = QGroupBox("Graphical Solution")
        graph_layout = QVBoxLayout(graph)
        self._canvas = FigureCanvas(Figure(figsize=(4, 3), facecolor="#111827"))
        self._canvas.setMinimumSize(190, 105)
        graph_layout.addWidget(self._canvas)
        self._lower.addWidget(graph)
        self._lower.setSizes((540, 300))
        vertical.addWidget(self._lower)
        vertical.setSizes((215, 410))
        self._main_stack = QStackedWidget()
        self._lesson_pages = {}
        self._main_stack.addWidget(vertical)
        self._hull_workspace = HullWorkspace()
        self._main_stack.addWidget(self._hull_workspace)
        outer.addWidget(self._main_stack, stretch=1)
        self._hull_controls.mode_changed.connect(self._convert)
        self._hull_controls.example_requested.connect(self._load_hull_example)
        self._hull_controls.ranges_changed.connect(self._hull_workspace.set_ranges)
        self._hull_controls.manual_cut_requested.connect(self._add_hull_cut)
        self._hull_workspace.status_changed.connect(self._status_label.setText)
        self._hull_workspace.manual_access_changed.connect(self._hull_controls.show_cut_access)

        self._mode_combo.currentIndexChanged.connect(self._convert)
        self._variable_count_combo.currentIndexChanged.connect(self._handle_count_change)
        self._constraint_count_combo.currentIndexChanged.connect(self._handle_count_change)
        self._objective_sense_combo.currentIndexChanged.connect(self._convert)

    def _add_input_sections(self, layout: QVBoxLayout) -> None:
        self._hull_controls = HullControls()
        self._hull_controls.hide()
        layout.insertWidget(0, self._hull_controls)
        self._branch_controls = QWidget()
        controls = QGridLayout(self._branch_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self._limit_label = QLabel("Node limit")
        self._node_limit = QSpinBox()
        self._node_limit.setRange(1, 1000)
        self._node_limit.setValue(DEFAULT_NODE_LIMIT)
        self._node_limit.setToolTip("Maximum number of LP relaxations solved.")
        controls.addWidget(self._limit_label, 0, 0)
        controls.addWidget(self._node_limit, 0, 1)
        self._incumbent_label = QLabel("Initial solution x₀")
        self._incumbent_edit = QLineEdit()
        self._incumbent_edit.setAccessibleName("Optional initial solution x₀")
        self._incumbent_edit.setPlaceholderText("Optional, e.g. 3, 2")
        self._incumbent_edit.setToolTip("Supply one coordinate per variable to seed the incumbent.")
        self._incumbent_label.setBuddy(self._incumbent_edit)
        self._incumbent_edit.returnPressed.connect(self._convert)
        controls.addWidget(self._incumbent_label, 1, 0)
        controls.addWidget(self._incumbent_edit, 1, 1)
        layout.addWidget(self._branch_controls)

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Calculation Steps")
        outer = QVBoxLayout(group)
        self._output_stack = QStackedWidget()
        outer.addWidget(self._output_stack)
        self._branch_steps = QWidget()
        layout = QVBoxLayout(self._branch_steps)
        layout.setContentsMargins(0, 0, 0, 0)
        self._calculations = self._browser()
        layout.addWidget(self._calculations, stretch=1)
        self._trace_toggle = QToolButton()
        self._trace_toggle.setText("Simplex intermediate calculations")
        self._trace_toggle.setCheckable(True)
        self._trace_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(self._trace_toggle)
        self._trace = self._browser()
        layout.addWidget(self._trace, stretch=1)
        self._trace_toggle.toggled.connect(self._update_trace)
        self._update_trace(False)
        self._output_stack.addWidget(self._branch_steps)
        self._relaxation_view = RelaxationSteps()
        self._relaxation_view.coordinates_changed.connect(self._refresh_relaxation_comparison)
        self._output_stack.addWidget(self._relaxation_view)
        return group

    def _browser(self) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setObjectName("stepsEditor")
        browser.setMinimumHeight(35)
        self._set_document_font(browser)
        return browser

    def _create_tree_group(self) -> QGroupBox:
        group = QGroupBox("Branch-and-Bound Tree")
        layout = QVBoxLayout(group)
        layout.setSpacing(5)
        self._tree = BranchTreeView()
        self._tree.setMinimumHeight(100)
        self._tree.node_selected.connect(self._select_node)
        layout.addWidget(self._tree, stretch=1)

        legend = QLabel(" &nbsp; ".join(
            f"<span style='color:{color}'>■</span> {status.value}" for status, color in COLORS.items()
        ))
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size: 12px; font-weight: normal;")
        layout.addWidget(legend)
        navigation = QGridLayout()
        self._previous = QPushButton("Previous Step")
        self._next = QPushButton("Next Step")
        self._final = QPushButton("Show Final Result")
        self._reset = QPushButton("Reset View")
        for i, (button, action) in enumerate((
            (self._previous, self._previous_step), (self._next, self._next_step),
            (self._final, self._show_final), (self._reset, self._reset_view),
        )):
            button.clicked.connect(action)
            navigation.addWidget(button, i // 2, i % 2)
        layout.addLayout(navigation)
        zoom = QHBoxLayout()
        for text, action in (
            ("Zoom In", lambda: self._tree.zoom(1.25)),
            ("Zoom Out", lambda: self._tree.zoom(.8)),
            ("Fit to View", self._tree.fit_tree),
        ):
            button = QPushButton(text)
            button.clicked.connect(action)
            zoom.addWidget(button)
        layout.addLayout(zoom)
        return group

    def _convert(self) -> None:
        mode = self._mode_combo.currentData()
        if mode in self._LESSONS:
            self._show_lesson(mode)
            return
        use_hull = mode == "hull"
        use_tree = mode == "bb"
        self._switch_workspace(use_hull, use_tree)
        if use_hull:
            self._convert_hull()
            return
        target = self._branch_steps if use_tree else self._relaxation_view
        mode_changed = target is not self._output_stack.currentWidget()
        self._output_stack.setCurrentWidget(target)
        if not use_tree:
            self._relaxation_view.set_variables(tuple(f"x{i + 1}" for i in range(len(self._domain_combos))))
        self._tree_group.setVisible(use_tree)
        if mode_changed:
            self._vertical_splitter.setSizes((215, 410) if use_tree else (360, 260))
        self._run = self._frame = self._lp_result = None
        self._tree.show_nodes({}, 0)
        for button in (self._previous, self._next, self._final, self._reset):
            button.setEnabled(False)
        try:
            raw, self._domains = self._read_problem()
            self._raw_problem = raw
            self._problem = relaxation(raw, self._domains)
            self._region = prepare_region(self._problem)
            if use_tree:
                seed_text = self._incumbent_edit.text().strip()
                seed = parse_vector(seed_text, "Initial solution x₀") if seed_text else None
                self._run = branch_and_bound(
                    raw, self._domains, node_limit=self._node_limit.value(), initial_incumbent=seed,
                )
            else:
                self._lp_result = solve_simplex(self._problem, capture_details=True)
        except (ValueError, ArithmeticError) as error:
            self._status_label.setText(f"Input or calculation error: {error}")
            self._calculations.clear()
            self._relaxation_view.clear_result()
            self._update_trace(self._trace_toggle.isChecked())
            plot_simplex_message(self._canvas.figure, "Enter a valid linear program.")
            self._canvas.draw_idle()
            return

        if self._run is not None:
            self._show_step(0, fit=True)
        elif self._lp_result is not None:
            self._status_label.setText(self._lp_result.message)
            self._relaxation_view.show_lp_result(self._problem, self._lp_result)
            self._refresh_relaxation_comparison()

    def _read_problem(self):
        objective, matrix, rhs, relations, _ = self._build_input_text()
        domains = tuple(Domain(combo.currentData()) for combo in self._domain_combos)
        raw = build_linear_program(
            ProblemForm.GENERAL, self._objective_sense_combo.currentData(),
            objective, matrix, rhs, relations, ", ".join(">=0" for _ in domains),
        )
        return raw, domains

    def _show_lesson(self, mode: str) -> None:
        if mode not in self._lesson_pages:
            lesson = self._LESSONS[mode]()
            self._main_stack.addWidget(lesson)
            self._lesson_pages[mode] = lesson
        self._status_label.hide()
        self._main_stack.setCurrentWidget(self._lesson_pages[mode])
        self._convert_button.hide()

    def _switch_workspace(self, use_hull: bool, use_tree: bool) -> None:
        target = self._hull_workspace if use_hull else self._vertical_splitter
        workspace_changed = self._main_stack.currentWidget() != target
        if workspace_changed:
            splitter = self._hull_workspace if use_hull else self._standard_top
            if splitter.indexOf(self._input_scroll) < 0:
                splitter.insertWidget(0, self._input_scroll)
                splitter.setSizes((390, 330, 295) if use_hull else (420, 420))
            self._main_stack.setCurrentWidget(target)
            self._input_scroll.show()
            if use_hull and self._status_row.indexOf(self._convert_button) < 0:
                self._input_button_layout.removeWidget(self._convert_button)
                self._status_row.addWidget(self._convert_button)
            elif not use_hull and self._input_button_layout.indexOf(self._convert_button) < 0:
                self._status_row.removeWidget(self._convert_button)
                self._input_button_layout.insertWidget(0, self._convert_button)
            self._convert_button.show()
        self._status_label.show()
        points = use_hull and self._hull_controls.mode == "points"
        self._hull_controls.setVisible(use_hull)
        self._editor_fields.setVisible(not points)
        self._variable_hint.setVisible(not points)
        self._branch_controls.setVisible(use_tree)
        if use_hull or workspace_changed:
            margin = 4 if use_hull else 9
            self._input_scroll.widget().layout().setContentsMargins(margin, 9, margin, 9)
            for edit in (*self._objective_edits, *self._right_hand_side_edits,
                         *(edit for row in self._constraint_coefficient_edits for edit in row)):
                edit.setFixedWidth(40 if use_hull else 62)
            for combo in self._relation_combos:
                combo.setFixedWidth(52 if use_hull else 62)
            for combo in self._domain_combos:
                combo.setFixedWidth(100 if use_hull else 116)

    def _convert_hull(self) -> None:
        try:
            if self._hull_controls.mode == "points":
                self._hull_workspace.solve_points(self._hull_controls.read_points())
            else:
                problem, domains = self._read_problem()
                self._hull_workspace.solve_problem(
                    self._hull_controls.mode, problem, domains, self._hull_controls.limits(),
                )
        except (ValueError, ArithmeticError) as error:
            self._hull_workspace.clear(f"Input or calculation error: {error}")

    def _load_hull_example(self, name: str) -> None:
        if self._hull_controls.mode == "points":
            self._hull_controls.points.setPlainText(POINT_EXAMPLES[name])
        else:
            example = ILP_EXAMPLES[name]
            self._set_combo_value(self._objective_sense_combo, example.sense.value)
            with QSignalBlocker(self._variable_count_combo), QSignalBlocker(self._constraint_count_combo):
                self._variable_count_combo.setCurrentIndex(self._count_index(2))
                self._constraint_count_combo.setCurrentIndex(self._count_index(len(example.constraints)))
            self._rebuild_dynamic_fields()
            for edit, value in zip(self._objective_edits, example.objective):
                edit.setText(str(value))
            for i, (x, y, rhs) in enumerate(example.constraints):
                for edit, value in zip(self._constraint_coefficient_edits[i], (x, y)):
                    edit.setText(str(value))
                self._right_hand_side_edits[i].setText(str(rhs))
                self._set_combo_value(self._relation_combos[i], "<=")
            for combo, domain in zip(self._domain_combos, example.domains):
                self._set_combo_value(combo, domain.value)
        self._convert()

    def _add_hull_cut(self, text: str) -> None:
        try:
            self._hull_workspace.add_cut(text)
        except (ValueError, ArithmeticError) as error:
            self._hull_controls.show_cut_error(str(error))

    def _refresh_relaxation_comparison(self) -> None:
        if self._lp_result is None:
            return
        point, message = self._relaxation_view.read_point()
        comparison = check_integer_point(self._raw_problem, self._domains, point) if point is not None else None
        self._relaxation_view.show_comparison(self._problem, self._lp_result, comparison, message)
        plot_solution(self._canvas.figure, self._problem, self._region, self._domains,
                      simplex=self._lp_result, comparison=comparison)
        self._canvas.draw_idle()

    def _previous_step(self) -> None:
        self._show_step(self._step - 1)

    def _next_step(self) -> None:
        self._show_step(self._step + 1)

    def _show_final(self) -> None:
        if self._run is not None:
            self._show_step(len(self._run.events) - 1, fit=True)

    def _reset_view(self) -> None:
        self._show_step(0, fit=True)

    def _show_step(self, step: int, *, fit: bool = False) -> None:
        if self._run is None or not 0 <= step < len(self._run.events):
            return
        self._step = step
        self._frame = self._run.frame(step)
        event = self._frame.event
        self._tree.show_nodes(self._frame.nodes, event.focus)
        last = len(self._run.events) - 1
        self._previous.setEnabled(step > 0)
        self._next.setEnabled(step < last)
        self._final.setEnabled(step < last)
        self._reset.setEnabled(True)
        # Show the current index without exposing later node information.
        self._status_label.setText(f"Step {step + 1} · {event.text}")
        self._select_node(event.focus)
        if fit:
            self._tree.fit_tree()
        else:
            self._tree.center_on_node(event.focus)

    def _select_node(self, identifier: int) -> None:
        if self._frame is None or identifier not in self._frame.nodes:
            return
        self._selected = identifier
        node = self._frame.nodes[identifier]
        self._tree.highlight(identifier)
        self._calculations.setHtml(format_node(node, self._frame, self._domains))
        self._update_trace(self._trace_toggle.isChecked())
        plot_solution(self._canvas.figure, self._problem, self._region, self._domains,
                      frame=self._frame, selected=node)
        self._canvas.draw_idle()

    def _update_trace(self, expanded: bool) -> None:
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self._trace_toggle.setArrowType(arrow)
        self._trace.setVisible(expanded)
        if not expanded:
            return
        result = self._frame.nodes[self._selected].simplex if self._frame is not None else None
        if result is None:
            self._trace.setPlainText("No LP calculation is available at this step.")
        else:
            self._trace.setHtml(format_simplex_trace(result))
