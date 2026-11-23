"""GUI tests for initial topic selection and navigation."""

import os
import unittest

from matplotlib.quiver import Quiver

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPointF, Qt, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on the local environment
    QApplication = None  # type: ignore[assignment]

from gui.main_window import MainWindow
from gui.topic_registry import TOPIC_REGISTRY
from topics.network_optimization.models import Method


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_first_topic_is_selected_on_startup(self) -> None:
        self.assertEqual(self.window.current_topic_id, TOPIC_REGISTRY[0].identifier)
        first_button = self.window._buttons_by_topic_id[TOPIC_REGISTRY[0].identifier]
        self.assertTrue(first_button.isChecked())

    def test_navigation_selects_each_registered_topic(self) -> None:
        self.assertEqual(
            set(self.window._buttons_by_topic_id),
            {"linear_programming", "unconstrained_optimization", "simplex_method",
             "integer_programming", "aitken", "network_optimization"},
        )
        for topic in TOPIC_REGISTRY:
            self.window.show_topic(topic.identifier)
            self.assertEqual(self.window.current_topic_id, topic.identifier)
            self.assertTrue(self.window._buttons_by_topic_id[topic.identifier].isChecked())

    def test_navigation_toggle_gives_the_same_page_more_space(self) -> None:
        self.window.show_topic("network_optimization")
        self.window.show()
        self.application.processEvents()
        page = self.window._page_stack.currentWidget()
        original_width = page.width()
        original_position = self.window._navigation_toggle.pos()

        self.window._navigation_toggle.click()
        self.application.processEvents()
        self.assertTrue(self.window._navigation_content.isHidden())
        self.assertTrue(self.window._navigation_toggle.isVisible())
        self.assertEqual(self.window._navigation_toggle.pos(), original_position)
        self.assertEqual(self.window._navigation_toggle.accessibleName(), "Show navigation")
        self.assertIs(self.window._page_stack.currentWidget(), page)
        self.assertGreater(page.width(), original_width)

        self.window._navigation_toggle.click()
        self.application.processEvents()
        self.assertTrue(self.window._navigation_content.isVisible())
        self.assertTrue(self.window._buttons_by_topic_id["network_optimization"].isChecked())
        self.assertEqual(page.width(), original_width)

    def test_problem_formulation_rebuilds_fields_from_selected_counts(self) -> None:
        page = self.window._page_stack.currentWidget()
        page._variable_count_combo.setCurrentIndex(2)  # three variables
        page._constraint_count_combo.setCurrentIndex(0)  # one constraint
        self.application.processEvents()

        self.assertEqual(len(page._objective_edits), 3)
        self.assertEqual(len(page._constraint_coefficient_edits), 1)
        self.assertEqual(len(page._domain_combos), 3)
        self.assertIn("Converted successfully", page._status_label.text())

    def test_problem_formulation_restores_general_form_options_after_resize(self) -> None:
        page = self.window._page_stack.currentWidget()
        relation = page._relation_combos[0]
        relation.setCurrentIndex(relation.findData(">="))
        domain = page._domain_combos[0]
        domain.setCurrentIndex(domain.findData("free"))

        source = page._source_form_combo
        source.setCurrentIndex(source.findData("standard"))
        self.assertEqual(relation.currentData(), "=")
        self.assertEqual(domain.currentData(), ">=0")
        page._variable_count_combo.setCurrentIndex(2)
        source.setCurrentIndex(source.findData("general"))

        self.assertEqual(page._relation_combos[0].currentData(), ">=")
        self.assertEqual(page._domain_combos[0].currentData(), "free")
        self.assertIn("Converted successfully", page._status_label.text())

    def test_unconstrained_page_accepts_a_formula_directly(self) -> None:
        self.window.show_topic("unconstrained_optimization")
        page = self.window._page_stack.currentWidget()

        page._expression_edit.setText("x1**2 + x2**2")
        page._variables_edit.setText("x1, x2")
        page._start_point_edit.setText("3, 1")
        page._initial_step_edit.setText("0.25")
        page._run_optimization()

        self.assertFalse(hasattr(page, "_function_combo"))
        self.assertEqual(page._initial_step_edit.text(), "0.25")
        self.assertIn("β trials", page._backtracking_table.chain.toPlainText())
        table = page._backtracking_table.table
        self.assertEqual(table.item(table.rowCount() - 1, 8).text(), "Accepted")
        self.assertIn("Function and derivatives", page._details_browser.toPlainText())
        self.assertFalse(page._max_iterations_spin.isHidden())
        self.assertEqual(page._max_iterations_label.text(), "Number of iterations")

    def test_simplex_manual_start_controls_and_path(self) -> None:
        self.window.show_topic("simplex_method")
        page = self.window._page_stack.currentWidget()
        self.assertEqual(page._start_mode_combo.currentText(), "Automatic")
        self.assertTrue(page._start_fields.isHidden())
        page._start_mode_combo.setCurrentIndex(1)
        self.assertFalse(page._start_fields.isHidden())
        page._start_edits[1].setText("4")
        page._convert_button.click()
        calculations = page._calculations.toPlainText()
        self.assertIn("Starting point: (0, 4)", calculations)
        self.assertIn("Start basis B", calculations)
        axis = page._canvas.figure.axes[0]
        path = next(line for line in axis.lines if line.get_label() == "Simplex path")
        self.assertEqual((path.get_xdata()[0], path.get_ydata()[0]), (0, 4))

        page._start_edits[0].setText("1")
        page._start_edits[1].setText("1")
        page._convert_button.click()
        self.assertIn("not a vertex", page._status_label.text())
        axis = page._canvas.figure.axes[0]
        self.assertIn("Invalid start", axis.get_legend_handles_labels()[1])
        self.assertNotIn("Simplex path", axis.get_legend_handles_labels()[1])
        self.assertEqual([edit.text() for edit in page._start_edits], ["1", "1"])

        page._variable_count_combo.setCurrentIndex(2)
        self.assertEqual([edit.text() for edit in page._start_edits], ["1", "1", "0"])
        page._start_mode_combo.setCurrentIndex(0)
        self.assertTrue(page._start_fields.isHidden())
        page._start_mode_combo.setCurrentIndex(1)
        self.assertEqual([edit.text() for edit in page._start_edits], ["1", "1", "0"])

    def test_successive_halving_navigation_uses_requested_iterations_and_marks_convergence(self) -> None:
        self.window.show_topic("unconstrained_optimization")
        page = self.window._page_stack.currentWidget()
        page._method_combo.setCurrentIndex(page._method_combo.findData("gradient_halving"))
        self.assertEqual(page._max_iterations_spin.value(), 10)
        self.assertEqual(page._max_iterations_spin.minimum(), 1)
        self.assertEqual(page._max_halvings_spin.value(), 50)
        self.assertEqual(page._halving_factor_edit.text(), "0.5")
        page._expression_edit.setText("x**2 + 2*y**2 - 2*x - 4*y")
        page._variables_edit.setText("x, y")
        page._start_point_edit.setText("4, 3")
        page._run_button.click()
        self.assertIn("Stationary point reached", page._status_label.text())
        self.assertNotIn("Done", page._status_label.text())

        page._expression_edit.setText("x**2 + y**2")
        page._start_point_edit.setText("1, 0")
        page._initial_step_edit.setText("0.25")
        page._tolerance_edit.setText("0.75")
        page._max_iterations_spin.setValue(4)
        page._run_button.click()
        table = page._intermediate_browser.toPlainText()
        self.assertIn("Done", page._status_label.text())
        self.assertEqual(page._status_label.property("state"), "success")
        self.assertIn("(accepted)", page._backtracking_table.chain.toPlainText())
        self.assertEqual(table.count("Convergence reached"), 1)
        self.assertEqual(page._iteration_spin.maximum(), 4)
        page._fit_run_button.click()
        axis = page._figure.axes[0]
        arrows = [
            collection for collection in axis.collections
            if isinstance(collection, Quiver)
        ]
        self.assertEqual(sum(len(arrow.U) for arrow in arrows), 1)
        self.assertEqual(sorted(text.get_text() for text in axis.texts if text.get_text()),
                         ["0", "1", "2", "3", "4"])
        marker = next(item for item in axis.collections if item.get_label() == "Convergence reached")
        self.assertEqual(marker.get_offsets()[0].tolist(), [0.25, 0.0])

    def test_ilp_tree_playback_selects_node_and_updates_plot_and_trace(self) -> None:
        self.window.show_topic("integer_programming")
        self.window.show()
        self.application.processEvents()
        page = self.window._page_stack.currentWidget()
        self.assertEqual(set(page._tree._node_items), {1})
        self.assertIn("Node 1 · Open", page._calculations.toPlainText())
        labels = page._canvas.figure.axes[0].get_legend_handles_labels()[1]
        self.assertNotIn("Root LP optimum", labels)
        self.assertNotIn("Feasible integer points", labels)
        page._trace_toggle.click()
        self.assertIn("No LP calculation", page._trace.toPlainText())
        page._next.click()
        self.assertIn("40/3", page._calculations.toPlainText())
        self.assertIn("Integrality check: Pending", page._calculations.toPlainText())
        self.assertIn("Phase II", page._trace.toPlainText())
        page._next.click()
        self.assertIn("fractional", page._calculations.toPlainText())
        self.assertIn("Branching variable: —", page._calculations.toPlainText())
        page._next.click()
        self.assertIn("Upper: 40/3", page._calculations.toPlainText())
        page._next.click()
        self.assertIn("Branching variable: x1 = 8/3", page._calculations.toPlainText())
        self.assertEqual(set(page._tree._node_items), {1})
        page._next.click()
        self.assertEqual(set(page._tree._node_items), {1, 2, 3})
        page._select_node(2)
        labels = page._canvas.figure.axes[0].get_legend_handles_labels()[1]
        self.assertIn("Selected branch limits", labels)
        self.assertNotIn("Selected LP solution", labels)
        page._previous.click()
        self.assertEqual(set(page._tree._node_items), {1})
        self.assertEqual(page._final.text(), "Show Final Result")
        page._final.click()
        self.assertIn("Global optimum proven", page._status_label.text())
        item = page._tree._node_items[2]
        location = page._tree.mapFromScene(item.sceneBoundingRect().center())
        self.assertTrue(page._tree.viewport().rect().contains(location))
        QTest.mouseClick(page._tree.viewport(), Qt.MouseButton.LeftButton, pos=location)
        details = page._calculations.toPlainText()
        self.assertIn("Node 2 · Integer", details)
        self.assertIn("x1 ≤ 2", details)
        self.assertIn("LP solution: (2, 3)", details)
        axis = page._canvas.figure.axes[0]
        branch = next(line for line in axis.lines if line.get_label() == "Selected branch limits")
        self.assertEqual(tuple(branch.get_xdata()), (2, 2))
        markers = {artist.get_label(): artist for artist in axis.collections}
        self.assertIn("Feasible integer points", markers)
        self.assertEqual(markers["Selected LP solution"].get_offsets()[0].tolist(), [2, 3])
        self.assertEqual(markers["Optimal integer solution"].get_offsets()[0].tolist(), [3, 2])
        page._previous.click()
        self.assertNotIn("Optimal", [item.node.status.value for item in page._tree._node_items.values()])
        self.assertNotIn("Global optimum", page._calculations.toPlainText())
        page._convert_button.click()
        self.assertEqual(set(page._tree._node_items), {1})
        self.assertIn("No LP calculation", page._trace.toPlainText())
        self.assertNotIn("Incumbent", page._canvas.figure.axes[0].get_legend_handles_labels()[1])
        page._mode_combo.setCurrentIndex(0)
        self.assertTrue(page._tree_group.isHidden())
        self.assertIn("LP optimum", page._canvas.figure.axes[0].get_legend_handles_labels()[1])

    def test_ilp_optional_initial_solution_respects_playback(self) -> None:
        self.window.show_topic("integer_programming")
        page = self.window._page_stack.currentWidget()
        page._incumbent_edit.setText("4, 3")
        page._convert_button.click()
        self.assertIn("infeasible", page._status_label.text())
        self.assertFalse(page._next.isEnabled())
        page._incumbent_edit.setText("3, 2")
        page._convert_button.click()
        self.assertEqual(set(page._tree._node_items), {1})
        self.assertIn("Incumbent at this step: None", page._calculations.toPlainText())
        page._next.click()
        self.assertIn("Initial solution x₀ · (3, 2)", page._calculations.toPlainText())
        self.assertIn("Incumbent", page._canvas.figure.axes[0].get_legend_handles_labels()[1])
        self.assertEqual(set(page._tree._node_items), {1})
        page._final.click()
        self.assertIn("Global optimum: x = (3, 2)", page._calculations.toPlainText())

    def test_network_matrix_graph_and_playback_are_synchronized(self) -> None:
        self.window.show_topic("network_optimization")
        page = self.window._page_stack.currentWidget()
        self.assertEqual(page._method_combo.count(), 7)
        page._matrix.item(0, 0).setText("Zürich")
        self.assertEqual(page._matrix.horizontalHeaderItem(1).text(), "Zürich")
        self.assertEqual(page._start_combo.currentText(), "Zürich")
        page._matrix.item(0, 2).setText("0")
        self.assertEqual(page._matrix.item(1, 1).text(), "0")
        self.assertEqual(page._graph.weight((0, 1)), 0)
        self.assertIn((0, 1), page._network_view._edges)

        page._method_combo.setCurrentIndex(page._method_combo.findData(Method.BFS.value))
        page._target_combo.setCurrentIndex(page._target_combo.findData(4))
        page._run_button.click()
        self.assertEqual(page._step_index, 0)
        self.assertIn("Initialisierung", page._results.toPlainText())
        page._next.click()
        self.assertIn("Queue", page._results.toPlainText())
        color = page._network_view._edges[(0, 1)]._line.pen().color().name()
        self.assertEqual(color, "#57b7f4")
        page._previous.click()
        self.assertEqual(page._network_view._edges[(0, 1)]._line.pen().color().name(), "#8393ad")
        page._last.click()
        self.assertIn("Weg zum Ziel", page._results.toPlainText())
        page._first.click()
        self.assertNotIn("Weg zum Ziel", page._results.toPlainText())

        position = QPointF(90, 50)
        page._network_view._nodes[0].setPos(position)
        self.assertEqual(page._network_view._positions[0], position)
        page._matrix.item(0, 2).setText("4")
        self.assertEqual(page._network_view._nodes[0].pos(), position)
        self.assertFalse(page._steps)
        page._count_spin.setValue(6)
        self.assertEqual(page._matrix.rowCount(), 6)
        self.assertEqual(page._matrix.item(0, 0).text(), "Zürich")
        self.assertEqual(page._matrix.item(0, 2).text(), "4")
        page._clear_matrix()
        self.assertEqual(page._graph.edges(), ())
        self.assertEqual(page._matrix.item(0, 1).text(), "0")
        page._load_example()
        self.assertEqual(page._matrix.rowCount(), 5)

    def test_network_finishing_orders_follow_playback_from_f(self) -> None:
        self.window.show_topic("network_optimization")
        page = self.window._page_stack.currentWidget()
        # Extend the built-in A–E example with F–A in the test editor.
        page._count_spin.setValue(6)
        page._matrix.item(5, 1).setText("1")
        page._start_combo.setCurrentIndex(page._start_combo.findData(5))

        page._method_combo.setCurrentIndex(page._method_combo.findData(Method.DFS.value))
        page._run_button.click()
        for _ in range(5):
            page._next.click()
        text = page._results.toPlainText()
        self.assertIn("Schritt 6/", text)
        self.assertIn("Discovery Order: F → A → B → C → D → E", text)
        self.assertIn("Finishing Order: —", text)
        self.assertIn("Stack · Spitze rechts: F, A, B, C, D, E", text)
        page._next.click()
        self.assertIn("Finishing Order: E", page._results.toPlainText())
        page._previous.click()
        self.assertIn("Finishing Order: —", page._results.toPlainText())
        page._last.click()
        self.assertIn("Finishing Order: E → D → C → B → A → F", page._results.toPlainText())

        page._method_combo.setCurrentIndex(page._method_combo.findData(Method.BFS.value))
        page._run_button.click()
        self.assertIn("Discovery Order: F\n", page._results.toPlainText())
        self.assertIn("Processing / Finishing Order: —", page._results.toPlainText())
        page._next.click()
        page._next.click()
        text = page._results.toPlainText()
        self.assertIn("Discovery Order: F → A → B → C\n", text)
        self.assertIn("Processing / Finishing Order: F → A\n", text)
        self.assertIn("Queue · nächster Knoten links: B, C", text)
        self.assertIn("Suchbaumkanten:", text)
        page._previous.click()
        self.assertIn("Discovery Order: F → A\n", page._results.toPlainText())
        page._last.click()
        text = page._results.toPlainText()
        self.assertIn("Discovery Order: F → A → B → C → D → E", text)
        self.assertIn("Processing / Finishing Order: F → A → B → C → D → E", text)

    def test_network_directed_arrows_and_algorithm_validation(self) -> None:
        self.window.show_topic("network_optimization")
        page = self.window._page_stack.currentWidget()
        page._directed_check.setChecked(True)
        page._matrix.item(0, 2).setText("-1")
        self.assertNotEqual(page._matrix.item(1, 1).text(), "-1")
        self.assertIsNotNone(page._network_view._edges[(0, 1)]._arrow)
        self.assertNotEqual(page._network_view._edges[(0, 1)]._line.path(),
                            page._network_view._edges[(1, 0)]._line.path())
        page._method_combo.setCurrentIndex(page._method_combo.findData(Method.DIJKSTRA.value))
        self.assertFalse(page._run_button.isEnabled())
        self.assertIn("nichtnegative", page._status.text())
        page._method_combo.setCurrentIndex(page._method_combo.findData(Method.FLOYD.value))
        self.assertTrue(page._run_button.isEnabled())
        page._method_combo.setCurrentIndex(page._method_combo.findData(Method.KRUSKAL.value))
        self.assertFalse(page._run_button.isEnabled())
        self.assertIn("ungerichteten", page._status.text())
        page._directed_check.setChecked(False)
        self.assertTrue(page._run_button.isEnabled())
        self.assertEqual(page._matrix.item(1, 1).text(), "-1")
        page._matrix.item(0, 2).setText("NaN")
        self.assertFalse(page._run_button.isEnabled())
        self.assertIn("Ungültiges Gewicht", page._status.text())

    def test_lp_relaxation_rechecks_manual_point_and_plot_without_resolving(self) -> None:
        self.window.show_topic("integer_programming")
        page = self.window._page_stack.currentWidget()
        page._mode_combo.setCurrentIndex(0)
        view = page._relaxation_view
        self.assertIs(page._output_stack.currentWidget(), view)
        self.assertEqual([edit.text() for edit in view._point_edits], ["", ""])
        self.assertIn("Optimaler Eckpunkt", view._lp_summary.text())
        self.assertIn("40/3", view._lp_summary.text())
        result = page._lp_result
        view._point_edits[0].setText("3")
        view._point_edits[1].setText("2")
        self.assertIs(page._lp_result, result)
        self.assertIn("Zulässiger Punkt", view._point_summary.text())
        self.assertIn("13 ≤ z<sup>*</sup><sub>ILP</sub> ≤ 40/3", view._bounds_summary.text())
        axis = page._canvas.figure.axes[0]
        markers = {item.get_label(): item for item in axis.collections}
        self.assertEqual(markers["Manual point"].get_offsets()[0].tolist(), [3, 2])
        self.assertIn("(3, 2)", [item.get_text() for item in axis.texts])
        view._trace_toggle.setChecked(True)
        self.assertIn("Phase II", view._trace.toPlainText())

        view._point_edits[0].setText("4")
        self.assertIn("Unzulässiger Punkt", view._point_summary.text())
        self.assertIn("C2:", view._point_summary.text())
        self.assertIn("Lower Bound: offen", view._bounds_summary.text())
        axis = page._canvas.figure.axes[0]
        markers = {item.get_label(): item for item in axis.collections}
        self.assertEqual(markers["Invalid manual point"].get_offsets()[0].tolist(), [4, 2])
        self.assertNotIn("Manual point", markers)
        view._point_edits[0].setText("not a number")
        self.assertIn("gültige Zahl", view._point_summary.text())
        self.assertNotIn("Invalid manual point", page._canvas.figure.axes[0].get_legend_handles_labels()[1])

        page._variable_count_combo.setCurrentIndex(2)
        self.assertEqual([edit.text() for edit in view._point_edits], ["not a number", "2", ""])

    def test_aitken_navigation_updates_table_and_both_plots_when_count_changes(self) -> None:
        self.window.show_topic("aitken")
        page = self.window._page_stack.currentWidget()
        self.assertEqual(page._function_edit.text(), "cos(x)")
        self.assertEqual(page._start_edit.text(), "1")
        self.assertEqual(page._points_spin.value(), 10)
        self.assertEqual(page._points_spin.minimum(), 3)
        self.assertEqual(page._table.topLevelItemCount(), 8)
        first = page._table.topLevelItem(0)
        self.assertEqual(first.text(4), "0.72801036")
        first.setExpanded(True)
        self.assertTrue(first.isExpanded())
        self.assertIn("D = xⁱ − 2xⁱ⁻¹ + xⁱ⁻²", first.child(0).text(0))
        self.assertIn("Residual", first.child(0).text(0))

        for count in (3, 5, 20):
            with self.subTest(count=count):
                page._points_spin.setValue(count)
                self.assertEqual(page._table.topLevelItemCount(), count - 2)
                self.assertIn(f"{count} points", page._status_label.text())
                for canvas in (page._values_canvas, page._residuals_canvas):
                    ordinary, estimated = canvas.figure.axes[0].lines
                    self.assertEqual(list(ordinary.get_xdata()), list(range(count)))
                    self.assertEqual(list(estimated.get_xdata()), list(range(2, count)))

        page._function_edit.setText("x")
        page._function_edit.returnPressed.emit()
        invalid = page._table.topLevelItem(0)
        self.assertEqual(invalid.text(4), "Invalid")
        self.assertIn("numerically zero", invalid.child(0).text(0))
        self.assertEqual(page._residuals_canvas.figure.axes[0].get_yscale(), "linear")

    def test_newton_and_broyden_plot_and_table_show_the_same_steps(self) -> None:
        self.window.show_topic("unconstrained_optimization")
        page = self.window._page_stack.currentWidget()
        page._expression_edit.setText("x**2 + 2*y**2 - 2*x - 4*y")
        page._variables_edit.setText("x,y")
        page._start_point_edit.setText("4,3")
        for method, matrix_label in (
            ("newton", "Hessian H(xₖ) used"),
            ("broyden", "Inverse Hessian approximation B used"),
        ):
            with self.subTest(method=method):
                page._method_combo.setCurrentIndex(page._method_combo.findData(method))
                page._run_button.click()
                self.assertIn("strict local minimum", page._status_label.text())
                table = page._intermediate_browser.toPlainText()
                for text in ("Step details", "Current point xₖ = (4, 3)",
                             "New point xₖ₊₁ = (1, 1)", "New function value f(xₖ₊₁) = -3"):
                    self.assertIn(text, table)
                details = page._details_browser.toPlainText()
                self.assertIn(matrix_label, details)
                if method == "broyden":
                    self.assertIn("s = xₖ₊₁ − xₖ = (-3, -2)", details)
                    self.assertIn("y = ∇f(xₖ₊₁) − ∇f(xₖ) = (0, 0) − (6, 8) = (-6, -8)", details)
                path = next(line for line in page._figure.axes[0].lines if line.get_label() == "iteration path")
                self.assertEqual(path.get_xdata().tolist(), [4, 1])
                self.assertEqual(path.get_ydata().tolist(), [3, 1])

    def test_parabola_fitting_navigation_connects_iteration_rows_and_plot(self) -> None:
        self.window.show_topic("unconstrained_optimization")
        page = self.window._page_stack.currentWidget()
        page._method_combo.setCurrentIndex(page._method_combo.findData("gradient_parabola"))

        self.assertEqual(page._initial_step_edit.text(), "0.25")
        self.assertEqual(page._max_iterations_spin.value(), 100)
        page._expression_edit.setText("x**2 + 2*y**2 - 2*x - 4*y")
        page._variables_edit.setText("x, y")
        page._start_point_edit.setText("4, 3")
        page._run_button.click()
        self.assertIn("0.304878", page._intermediate_browser.toPlainText())
        self.assertIn("Selected β*", page._details_browser.toPlainText())
        self.assertTrue(any(isinstance(item, Quiver) for item in page._figure.axes[0].collections))

        page._intermediate_browser.anchorClicked.emit(QUrl("#parabola-2"))
        self.assertEqual(page._iteration_spin.value(), 2)
        self.assertIs(page._plot_tabs.currentWidget(), page._method_tab)
        self.assertIn("Iteration 2", page._details_browser.toPlainText())
        _, step = page._result.parabola_iterations[1]
        axes = page._method_figure.axes[0]
        fitted_marker = next(line for line in axes.lines if line.get_label() == "β*")
        self.assertAlmostEqual(fitted_marker.get_xdata()[0], step.fitted.step_size)

        page._iteration_spin.setValue(1)
        self.assertIn("Iteration 1", page._details_browser.toPlainText())

    def test_parabola_fitting_can_plot_a_one_variable_function(self) -> None:
        self.window.show_topic("unconstrained_optimization")
        page = self.window._page_stack.currentWidget()
        page._method_combo.setCurrentIndex(page._method_combo.findData("gradient_parabola"))
        page._expression_edit.setText("(x - 1)**2")
        page._variables_edit.setText("x")
        page._start_point_edit.setText("4")
        page._run_button.click()

        self.assertIn("Final point: (1)", page._status_label.text())
        self.assertEqual(page._iteration_spin.maximum(), 1)
        self.assertEqual(page._figure.axes[0].get_ylabel(), "f(x)")
        self.assertEqual(page._method_figure.axes[0].get_xlabel(), "Step size β")

