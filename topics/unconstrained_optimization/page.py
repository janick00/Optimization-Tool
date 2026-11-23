"""User interface for unconstrained continuous optimization."""

from __future__ import annotations

import re
from math import isfinite

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, QTimer, Qt, QUrl
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from topics.unconstrained_optimization.backtracking_table import BacktrackingTable
from topics.unconstrained_optimization.formatting import (
    format_intermediate_details_html,
    format_optimization_details_html,
)
from topics.unconstrained_optimization.history import IterationFrame, iteration_history
from topics.unconstrained_optimization.models import (
    OptimizationMethod,
    OptimizationResult,
    OptimizationSettings,
)
from topics.unconstrained_optimization.objective import parse_function
from topics.unconstrained_optimization.solver import optimize
from topics.unconstrained_optimization.step_plots import METHOD_TABS, plot_convergence, plot_method_step
from topics.unconstrained_optimization.visualization import (
    BACKGROUND,
    PLOT_LIMIT,
    plot_empty_state,
    plot_optimization_result,
    subplot_bottom,
)

DEMO_FUNCTION = "(x1 - 2)**4 / 16 + (x1 - 2)**2 / 2 + (x2 - 1)**2 / 2 + (x1 - 2*x2)**2 / 8"


class _PlotToolbar(NavigationToolbar2QT):
    toolitems = tuple(
        item for item in NavigationToolbar2QT.toolitems if item[0] in ("Home", "Back", "Forward", "Pan", "Zoom")
    )


class UnconstrainedOptimizationPage(QWidget):
    """Run a selected continuous-optimization method and inspect its history."""

    def __init__(
        self, parent: QWidget | None = None, *, method: OptimizationMethod | None = None
    ) -> None:
        super().__init__(parent)
        self._fixed_method = method
        self._input_group: str | None = None
        self._method_inputs: dict[str, tuple[str, int]] = {}
        self._result: OptimizationResult | None = None
        self._frames: tuple[IterationFrame, ...] = ()
        self._active_index = 0
        self._active_trial: int | None = None
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.setInterval(90)
        self._viewport_timer.timeout.connect(self._refresh_trajectory_view)
        self._build_ui()
        self._run_optimization()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(26, 18, 26, 18)
        outer_layout.setSpacing(7)

        title_label = QLabel({
            OptimizationMethod.GRADIENT_HALVING: "Successive Halving",
            OptimizationMethod.GRADIENT_PARABOLA: "Parabola Fitting",
        }.get(self._fixed_method, "Unconstrained Optimization"))
        title_label.setObjectName("topicTitle")
        title_label.setWordWrap(True)
        heading = QHBoxLayout()
        heading.addWidget(title_label, stretch=1)
        self._method_combo = QComboBox()
        self._method_combo.setAccessibleName("Optimization method")
        for method in (self._fixed_method,) if self._fixed_method else OptimizationMethod:
            self._method_combo.addItem(method.label, method.value)
        self._method_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._method_combo.setMaximumWidth(350)
        self._method_combo.currentTextChanged.connect(self._method_combo.setToolTip)
        self._method_combo.setToolTip(self._method_combo.currentText())
        self._method_combo.setVisible(self._fixed_method is None)
        heading.addWidget(self._method_combo)
        outer_layout.addLayout(heading)

        outer_layout.addWidget(self._create_setup_group())
        outer_layout.addLayout(self._create_navigation())
        outer_layout.addWidget(self._create_results_splitter(), stretch=1)
        self._method_combo.currentIndexChanged.connect(self._change_method)

    def _create_setup_group(self) -> QGroupBox:
        group = QGroupBox("Optimization setup")
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setVerticalSpacing(5)
        self._setup_layout = layout

        has_example = self._fixed_method is not None
        self._expression_edit = QLineEdit(
            "x**2 + 2*y**2 - 2*x - 4*y" if has_example else DEMO_FUNCTION
        )
        self._expression_edit.setPlaceholderText(f"e.g. {DEMO_FUNCTION}")
        self._expression_edit.setToolTip(
            "Enter a SymPy-compatible formula, for example x1**2 + sin(x2)."
        )
        layout.addRow("Function f(x)", self._expression_edit)

        variable_start_layout = QHBoxLayout()
        self._variables_edit = QLineEdit("x, y" if has_example else "x1, x2")
        self._variables_edit.setPlaceholderText("x1, x2")
        variable_start_layout.addWidget(self._variables_edit, stretch=1)
        variable_start_layout.addWidget(QLabel("Start point x0"))
        self._start_point_edit = QLineEdit("4, 3" if has_example else "0, 0")
        self._start_point_edit.setPlaceholderText("0, 0")
        variable_start_layout.addWidget(self._start_point_edit, stretch=1)
        layout.addRow("Variables", variable_start_layout)

        parameter_layout = QHBoxLayout()
        self._step_label = QLabel()
        parameter_layout.addWidget(self._step_label)
        self._initial_step_edit = QLineEdit()
        self._initial_step_edit.setFixedWidth(90)
        parameter_layout.addWidget(self._initial_step_edit)
        parameter_layout.addWidget(QLabel("Tolerance"))
        self._tolerance_edit = QLineEdit("1e-6")
        self._tolerance_edit.setFixedWidth(90)
        parameter_layout.addWidget(self._tolerance_edit)
        self._max_iterations_label = QLabel("Max. iterations")
        parameter_layout.addWidget(self._max_iterations_label)
        self._max_iterations_spin = QSpinBox()
        self._max_iterations_spin.setRange(1, 1000)
        self._max_iterations_spin.setValue(OptimizationSettings.max_iterations)
        self._max_iterations_spin.setFixedWidth(90)
        parameter_layout.addWidget(self._max_iterations_spin)
        parameter_layout.addStretch()
        layout.addRow("Parameters", parameter_layout)

        self._halving_parameters = QWidget()
        halving_layout = QHBoxLayout(self._halving_parameters)
        halving_layout.setContentsMargins(0, 0, 0, 0)
        halving_layout.addWidget(QLabel("Halving factor"))
        self._halving_factor_edit = QLineEdit(str(OptimizationSettings.halving_factor))
        self._halving_factor_edit.setFixedWidth(90)
        self._halving_factor_edit.setToolTip("Multiply β by this factor after a rejected trial.")
        halving_layout.addWidget(self._halving_factor_edit)
        halving_layout.addWidget(QLabel("Max. halvings"))
        self._max_halvings_spin = QSpinBox()
        self._max_halvings_spin.setRange(0, 1000)
        self._max_halvings_spin.setValue(OptimizationSettings.max_halvings)
        self._max_halvings_spin.setFixedWidth(90)
        self._max_halvings_spin.setToolTip("Allowed reductions after the initial β₀ trial.")
        halving_layout.addWidget(self._max_halvings_spin)
        halving_layout.addStretch()
        layout.addRow("Backtracking", self._halving_parameters)

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
        self._update_step_input()
        return group

    def _create_navigation(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._iteration_spin = QSpinBox()
        self._iteration_spin.setAccessibleName("Optimization iteration")
        self._iteration_spin.setKeyboardTracking(False)
        self._iteration_spin.valueChanged.connect(self._select_iteration)
        layout.addWidget(QLabel("Iteration"))
        layout.addWidget(self._iteration_spin)
        self._iteration_total = QLabel()
        layout.addWidget(self._iteration_total)
        self._previous_button = self._navigation_button("Previous", "Previous iteration", lambda: self._move_iteration(-1))
        self._next_button = self._navigation_button("Next", "Next iteration", lambda: self._move_iteration(1))
        self._reset_button = self._navigation_button("Reset", "Reset iteration and plot", self._reset_iteration)
        for button in (self._previous_button, self._next_button, self._reset_button):
            layout.addWidget(button)
        layout.addStretch()
        self._focus_step = QCheckBox("Auto-fit step")
        self._focus_step.setChecked(True)
        self._focus_step.setToolTip("Fit the current, next and selected trial points when changing steps.")
        self._focus_step.toggled.connect(self._fit_trajectory)
        layout.addWidget(self._focus_step)
        self._fit_run_button = self._navigation_button("Fit run", "Fit full trajectory", self._fit_run)
        layout.addWidget(self._fit_run_button)
        return layout

    @staticmethod
    def _navigation_button(text, name, action) -> QPushButton:
        button = QPushButton(text)
        button.setAccessibleName(name)
        button.clicked.connect(action)
        return button

    def _create_results_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        plot_group = QGroupBox("Plots")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(5, 6, 5, 5)
        self._plot_tabs = QTabWidget()
        self._plot_tabs.setDocumentMode(True)
        plot_layout.addWidget(self._plot_tabs)
        _, self._canvas, self._trajectory_toolbar = self._create_plot_tab("Trajectory")
        self._figure = self._canvas.figure
        self._method_tab, self._method_canvas, self._method_toolbar = self._create_plot_tab("")
        self._method_figure = self._method_canvas.figure
        _, self._convergence_canvas, self._convergence_toolbar = self._create_plot_tab("Convergence")
        self._canvas.mpl_connect("pick_event", self._pick_iteration)
        self._method_canvas.mpl_connect("pick_event", self._pick_trial)
        splitter.addWidget(plot_group)

        result_group = QGroupBox("Intermediate results")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(5, 6, 5, 5)
        panes = QSplitter(Qt.Orientation.Vertical)
        panes.setChildrenCollapsible(False)
        self._result_tabs = QTabWidget()
        self._result_tabs.setDocumentMode(True)
        self._result_tabs.setTabBarAutoHide(True)
        self._intermediate_browser = self._calculation_browser()
        self._trial_browser = self._calculation_browser()
        for browser in (self._intermediate_browser, self._trial_browser):
            browser.setOpenLinks(False)
            browser.anchorClicked.connect(self._select_reference)
        self._trial_panel = QStackedWidget()
        self._backtracking_table = BacktrackingTable()
        self._backtracking_table.trial_selected.connect(self._select_backtracking_trial)
        self._trial_panel.addWidget(self._backtracking_table)
        self._trial_panel.addWidget(self._trial_browser)
        self._result_tabs.addTab(self._intermediate_browser, "Iterations")
        self._result_tabs.addTab(self._trial_panel, "Backtracking")
        panes.addWidget(self._result_tabs)

        math_panel = QWidget()
        math_layout = QVBoxLayout(math_panel)
        math_layout.setContentsMargins(0, 0, 0, 0)
        math_layout.setSpacing(3)
        details_label = QLabel("Calculations and formulas")
        details_label.setObjectName("sectionLabel")
        math_layout.addWidget(details_label)
        self._details_browser = self._calculation_browser()
        math_layout.addWidget(self._details_browser, stretch=1)
        panes.addWidget(math_panel)
        panes.setSizes((197, 165))
        result_layout.addWidget(panes)
        splitter.addWidget(result_group)
        splitter.setSizes((495, 430))
        return splitter

    def _create_plot_tab(self, title):
        tab = QWidget(self._plot_tabs)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(3)
        canvas = FigureCanvas(Figure(figsize=(4.5, 3.5), facecolor=BACKGROUND))
        canvas.setMinimumSize(240, 125)
        canvas.mpl_connect("resize_event", self._layout_resized_plot)
        toolbar = _PlotToolbar(canvas, tab, coordinates=False)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, stretch=1)
        self._plot_tabs.addTab(tab, title)
        return tab, canvas, toolbar

    @staticmethod
    def _layout_resized_plot(event) -> None:
        figure = event.canvas.figure
        figure.subplots_adjust(bottom=subplot_bottom(figure))

    @staticmethod
    def _calculation_browser() -> QTextBrowser:
        browser = QTextBrowser()
        browser.setObjectName("optimizationDetails")
        browser.setReadOnly(True)
        browser.setMinimumHeight(60)
        font = QFont("DejaVu Serif")
        font.setStyleHint(QFont.StyleHint.Serif)
        font.setPointSize(9)
        browser.setFont(font)
        return browser

    def _update_step_input(self) -> None:
        method = OptimizationMethod(self._method_combo.currentData())
        uses_halving = method is OptimizationMethod.GRADIENT_HALVING
        input_group = method.value
        if self._input_group != input_group:
            if self._input_group is not None:
                self._method_inputs[self._input_group] = (
                    self._initial_step_edit.text(), self._max_iterations_spin.value()
                )
            defaults = OptimizationSettings.for_method(method)
            beta, iterations = self._method_inputs.get(
                input_group, (str(defaults.initial_step), defaults.max_iterations)
            )
            self._initial_step_edit.setText(beta)
            self._max_iterations_spin.setValue(iterations)
            self._input_group = input_group
        iterations_label = "Number of iterations" if uses_halving else "Max. iterations"
        self._max_iterations_label.setText(iterations_label)
        self._max_iterations_spin.setAccessibleName(iterations_label)
        self._setup_layout.setRowVisible(self._halving_parameters, uses_halving)
        enabled = method not in (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN)
        self._step_label.setText("Initial step β₀")
        self._step_label.setVisible(enabled)
        self._initial_step_edit.setVisible(enabled)
        self._initial_step_edit.setEnabled(enabled)
        self._initial_step_edit.setToolTip({
            OptimizationMethod.GRADIENT_HALVING: "First trial size. Each later iteration keeps the last accepted β.",
            OptimizationMethod.GRADIENT_PARABOLA: "Initial β for every iteration. Samples use 0, β and 2β.",
        }.get(method, ""))

    def _change_method(self) -> None:
        self._update_step_input()
        self._run_optimization()

    def _run_optimization(self) -> None:
        try:
            function = parse_function(
                self._expression_edit.text(),
                self._variables_edit.text(),
            )
            start = _parse_start_point(self._start_point_edit.text())
            method = OptimizationMethod(self._method_combo.currentData())
            halving_settings = (
                {
                    "halving_factor": float(self._halving_factor_edit.text()),
                    "max_halvings": self._max_halvings_spin.value(),
                }
                if method is OptimizationMethod.GRADIENT_HALVING else {}
            )
            settings = OptimizationSettings(
                initial_step=(
                    float(self._initial_step_edit.text())
                    if self._initial_step_edit.isEnabled()
                    else OptimizationSettings.initial_step
                ),
                tolerance=float(self._tolerance_edit.text()),
                max_iterations=self._max_iterations_spin.value(),
                **halving_settings,
            )
            result = optimize(function, method, start, settings)
        except (ValueError, TypeError, OverflowError) as error:
            self._show_error(str(error))
            return

        self._show_result(result)

    def _show_result(self, result: OptimizationResult) -> None:
        self._viewport_timer.stop()
        method_changed = self._result is None or self._result.method is not result.method
        self._result = result
        self._frames = iteration_history(result)
        self._set_method_tabs(result.method, choose_default=method_changed)
        self._intermediate_browser.setToolTip("Select an iteration number to inspect that transition.")
        self._trial_browser.setToolTip("Select a trial to see its position and numeric substitution.")
        with QSignalBlocker(self._iteration_spin):
            self._iteration_spin.setRange(self._frames[0].number, self._frames[-1].number)
            self._iteration_spin.setValue(self._frames[0].number)
        self._show_iteration(0, refit=True)
        accepted = result.accepted_halving_step
        if result.method is OptimizationMethod.GRADIENT_HALVING:
            point_label = "Last accepted point" if accepted else "Start point"
            success = result.completed_requested_iterations
        else:
            point_label = "Final point"
            success = result.converged
        message = result.message
        if result.method is OptimizationMethod.GRADIENT_HALVING and not message.endswith("."):
            message += "."
        self._status_label.setText(
            f"{message} {point_label}: {self._format_point(result.final_point)}"
        )
        self._status_label.setProperty(
            "state", "success" if success else "warning"
        )
        self._refresh_style(self._status_label)

    def _select_iteration(self, number: int) -> None:
        if self._frames:
            index = number - self._frames[0].number
            if 0 <= index < len(self._frames):
                self._show_iteration(index)

    def _show_iteration(self, index: int, *, refit=False) -> None:
        self._viewport_timer.stop()
        frame = self._frames[index]
        self._active_index = index
        self._active_trial = frame.chosen_trial
        if self._active_trial is None and frame.trial_points:
            self._active_trial = 0
        with QSignalBlocker(self._iteration_spin):
            self._iteration_spin.setValue(frame.number)
        self._sync_navigation()
        self._render_iteration(refit=refit)
        self._convergence_toolbar.update()
        plot_convergence(self._convergence_canvas.figure, self._result, frame)
        self._convergence_canvas.draw_idle()

    def _render_iteration(self, *, refit=False) -> None:
        frame = self._frames[self._active_index]
        self._intermediate_browser.setHtml(format_intermediate_details_html(
            self._result, frames=self._frames, active=self._active_index,
        ))
        if self._result.method is OptimizationMethod.GRADIENT_HALVING:
            self._backtracking_table.show_frame(frame, self._active_trial)
        elif self._result.method is OptimizationMethod.GRADIENT_PARABOLA:
            self._trial_browser.setHtml(format_intermediate_details_html(
                self._result, frames=self._frames, active=self._active_index, selected_trial=self._active_trial,
                trials_only=True,
            ))
        self._details_browser.setHtml(format_optimization_details_html(
            self._result, frame=frame, selected_trial=self._active_trial,
        ))
        if refit or self._focus_step.isChecked():
            self._fit_trajectory()
        else:
            self._draw_trajectory(self._view_limits())
        self._method_toolbar.update()
        plot_method_step(self._method_figure, self._result, frame, self._active_trial)
        self._method_canvas.draw_idle()

    def _move_iteration(self, delta: int) -> None:
        index = self._active_index + delta
        if 0 <= index < len(self._frames):
            self._show_iteration(index)

    def _reset_iteration(self) -> None:
        if self._frames:
            self._show_iteration(0, refit=True)

    def _sync_navigation(self) -> None:
        count = len(self._frames)
        self._previous_button.setEnabled(self._active_index > 0 and count > 1)
        self._next_button.setEnabled(self._active_index + 1 < count)
        self._iteration_spin.setEnabled(count > 1)
        self._iteration_total.setText(f"/ {self._frames[-1].number}" if count else "")
        for field in (self._reset_button, self._fit_run_button, self._focus_step):
            field.setEnabled(count > 0)

    def _select_reference(self, link: QUrl) -> None:
        match = re.fullmatch(r"(iteration|trial|parabola)-(\d+)", link.fragment())
        if match and self._frames:
            target, number = match.groups()
            if target == "trial":
                self._select_trial(int(number))
                self._plot_tabs.setCurrentWidget(self._method_tab)
            else:
                self._select_iteration(int(number))
                if target == "parabola":
                    self._plot_tabs.setCurrentWidget(self._method_tab)

    def _select_trial(self, index: int) -> None:
        if self._frames and 0 <= index < len(self._frames[self._active_index].trial_points):
            if index != self._active_trial:
                self._active_trial = index
                self._render_iteration()
            self._result_tabs.setCurrentWidget(self._trial_panel)

    def _select_backtracking_trial(self, index: int) -> None:
        if not self._frames or not 0 <= index < len(self._frames[self._active_index].trials):
            return
        self._select_trial(index)
        self._plot_tabs.setCurrentIndex(0)
        if self._focus_step.isChecked():
            return
        trial = self._frames[self._active_index].trials[index]
        coordinates = (trial.point[0], trial.value) if len(trial.point) == 1 else trial.point[:2]
        view = self._view_limits()
        if view and any(
            value is not None and isfinite(value) and abs(value) < PLOT_LIMIT
            and not min(limits) <= value <= max(limits)
            for value, limits in zip(coordinates, view)
        ):
            self._draw_trajectory(focus=True)

    def _set_method_tabs(self, method: OptimizationMethod, *, choose_default: bool) -> None:
        self._plot_tabs.setTabText(1, METHOD_TABS[method])
        halving = method is OptimizationMethod.GRADIENT_HALVING
        gradient_method = halving or method is OptimizationMethod.GRADIENT_PARABOLA
        self._trial_panel.setCurrentWidget(self._backtracking_table if halving else self._trial_browser)
        self._result_tabs.setTabVisible(1, gradient_method)
        self._result_tabs.tabBar().setVisible(gradient_method)
        self._result_tabs.setTabText(1, "Backtracking" if halving else "Samples")
        if choose_default:
            self._result_tabs.setCurrentIndex(1 if gradient_method else 0)

    def _pick_iteration(self, event) -> None:
        if not self._trajectory_toolbar.mode and event.artist.get_gid() == "iterations" and len(event.ind):
            self._select_iteration(min(int(event.ind[0]) + 1, self._frames[-1].number))

    def _pick_trial(self, event) -> None:
        group = event.artist.get_gid()
        if not self._method_toolbar.mode and group and group.startswith("trials:") and len(event.ind):
            indices = group.partition(":")[2].split(",")
            self._select_trial(int(indices[event.ind[0]]))

    def _view_limits(self):
        if self._figure.axes and self._figure.axes[0].axison:
            axis = self._figure.axes[0]
            return axis.get_xlim(), axis.get_ylim()
        return None

    def _fit_run(self) -> None:
        with QSignalBlocker(self._focus_step):
            self._focus_step.setChecked(False)
        self._fit_trajectory()

    def _fit_trajectory(self, _checked=False) -> None:
        self._viewport_timer.stop()
        self._trajectory_toolbar.update()
        self._draw_trajectory()

    def _queue_trajectory_refresh(self, _axis) -> None:
        self._viewport_timer.start()

    def _refresh_trajectory_view(self) -> None:
        if self._frames:
            view = self._view_limits()
            if view and all(-PLOT_LIMIT < x < PLOT_LIMIT for limits in view for x in limits):
                self._draw_trajectory(view)

    def _draw_trajectory(self, view=None, *, focus: bool | None = None) -> None:
        if not self._frames:
            return
        axis = plot_optimization_result(
            self._figure, self._result, self._frames[self._active_index],
            selected_trial=self._active_trial,
            focus=self._focus_step.isChecked() if focus is None else focus, view=view,
        )
        axis.callbacks.connect("xlim_changed", self._queue_trajectory_refresh)
        axis.callbacks.connect("ylim_changed", self._queue_trajectory_refresh)
        self._canvas.draw_idle()

    def _show_error(self, message: str) -> None:
        self._viewport_timer.stop()
        self._result = None
        self._frames = ()
        self._active_index = 0
        self._sync_navigation()
        method = OptimizationMethod(self._method_combo.currentData())
        self._set_method_tabs(method, choose_default=True)
        self._status_label.setText(f"Input or calculation error: {message}")
        self._status_label.setProperty("state", "error")
        self._intermediate_browser.clear()
        self._trial_browser.clear()
        self._backtracking_table.show_frame(None, None)
        self._details_browser.clear()
        for canvas in (self._canvas, self._method_canvas, self._convergence_canvas):
            plot_empty_state(canvas.figure)
            canvas.draw_idle()
        self._refresh_style(self._status_label)

    @staticmethod
    def _format_point(point: tuple[float, ...]) -> str:
        return "(" + ", ".join(f"{value:.6g}" for value in point) + ")"

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _parse_start_point(text: str) -> tuple[float, ...]:
    tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
    if not tokens:
        raise ValueError("Enter a starting point, for example 0, 0.")
    return tuple(float(token) for token in tokens)
