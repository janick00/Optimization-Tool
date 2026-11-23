"""Check the four playback views against their numerical iteration records."""

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from topics.unconstrained_optimization.history import iteration_history
from topics.unconstrained_optimization.models import OptimizationMethod, OptimizationSettings
from topics.unconstrained_optimization.objective import ParsedFunction, parse_function
from topics.unconstrained_optimization.page import UnconstrainedOptimizationPage
from topics.unconstrained_optimization.solver import optimize
from topics.unconstrained_optimization.step_plots import quadratic_model


QUADRATIC = "x**2 + 2*y**2 - 2*x - 4*y"


class PlaybackCalculationTests(TestCase):
    def test_newton_model_uses_recorded_gradient_hessian_and_full_step(self):
        result = optimize(parse_function(QUADRATIC, "x,y"), OptimizationMethod.NEWTON, (4, 3))
        frame, = iteration_history(result)
        details = frame.second_order
        self.assertEqual(frame.beta, 1.0)
        self.assertEqual(frame.direction, (-3., -2.))
        self.assertAlmostEqual(frame.following.step_size, np.sqrt(13))
        np.testing.assert_allclose(np.asarray(details.matrix) @ frame.direction, -np.asarray(frame.current.gradient))
        self.assertEqual(quadratic_model(frame.direction, frame.current.value,
                                         frame.current.gradient, details.matrix), -3)
        self.assertEqual(result.convergence_iteration, 1)

    def test_broyden_trace_exposes_the_applied_rank_one_update(self):
        result = optimize(parse_function("(x-1)**4 + (y+2)**2", "x,y"), OptimizationMethod.BROYDEN, (2, 0))
        first, second, *_ = iteration_history(result)
        details = first.second_order
        update = details.update
        matrix = np.asarray(details.matrix)
        np.testing.assert_allclose(matrix @ (-np.asarray(first.current.gradient)), first.direction)
        np.testing.assert_allclose(np.asarray(first.current.point) + first.beta * np.asarray(first.direction),
                                   first.following.point)
        np.testing.assert_allclose(np.outer(update.discrepancy, update.row) / update.denominator, update.correction)
        np.testing.assert_allclose(matrix - update.correction, second.second_order.matrix)
        np.testing.assert_allclose(np.asarray(update.matrix) @ details.gradient_change, details.displacement)
        np.testing.assert_allclose(update.row @ np.asarray(update.scaled_gradient_change), update.denominator)
        self.assertIsNone(result.steps[-1].second_order.update)
        self.assertIn("gradient tolerance", result.steps[-1].second_order.update_note)

    def test_history_keeps_failed_trials_without_creating_a_new_point(self):
        settings = OptimizationSettings(initial_step=1., max_halvings=0)
        result = optimize(parse_function("x**2", "x"), OptimizationMethod.GRADIENT_HALVING, (1,), settings)
        frame, = iteration_history(result)
        self.assertEqual(frame.number, 1)
        self.assertEqual(len(result.steps), 1)
        self.assertIsNone(frame.following)
        self.assertEqual(frame.trial_points, result.failed_trials)
        self.assertEqual(frame.trial_decision(0), "Rejected")


class PlaybackWidgetTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = UnconstrainedOptimizationPage()
        self.page.resize(1050, 700)
        self.page.show()
        self.addCleanup(self._close)

    def _close(self):
        self.page.close()
        self.page.deleteLater()
        self.application.processEvents()

    def _run(self, method, expression=QUADRATIC, variables="x,y", start="4,3"):
        page = self.page
        page._expression_edit.setText(expression)
        page._variables_edit.setText(variables)
        page._start_point_edit.setText(start)
        index = page._method_combo.findData(method)
        if index == page._method_combo.currentIndex():
            page._run_button.click()
        else:
            page._method_combo.setCurrentIndex(index)
        return page._frames[0]

    def _trial_markers(self):
        markers = {}
        for collection in self.page._method_figure.axes[0].collections:
            group = collection.get_gid() or ""
            if group.startswith("trials:"):
                indices = map(int, group.partition(":")[2].split(","))
                markers.update(zip(indices, collection.get_offsets()))
        return markers

    def test_halving_trial_selection_updates_plot_table_and_calculation(self):
        page = self.page
        frame = self._run("gradient_halving")
        self.assertEqual(page._plot_tabs.tabText(1), "Line search")
        self.assertEqual(page._iteration_spin.maximum(), 2)
        self.assertEqual(page._active_trial, frame.chosen_trial)
        markers = self._trial_markers()
        self.assertEqual(set(markers), {0, 1})
        for index, trial in enumerate(frame.trials):
            np.testing.assert_allclose(markers[index], [trial.step_size, trial.value])
        page._plot_tabs.setCurrentWidget(page._method_tab)
        table = page._backtracking_table.table
        table.setCurrentCell(0, 1)
        self.assertEqual(page._active_trial, 0)
        self.assertEqual(page._plot_tabs.currentIndex(), 0)
        details = page._details_browser.toPlainText()
        self.assertIn("Selected trial 1 · Rejected", details)
        self.assertIn("x_trial = xₖ + β · dₖ", details)
        self.assertIn("φ(β) = f(x_trial) = 78 > 14", details)
        marker = next(m for m in page._figure.axes[0].collections if m.get_label() == "Selected trial")
        np.testing.assert_allclose(marker.get_offsets()[0], [-2, -5])
        accepted = next(m for m in page._method_figure.axes[0].collections if m.get_label() == "Accepted")
        page._pick_trial(SimpleNamespace(artist=accepted, ind=[0]))
        self.assertEqual(page._active_trial, frame.chosen_trial)
        self.assertEqual(table.currentRow(), frame.chosen_trial)
        marker = next(m for m in page._figure.axes[0].collections if m.get_label() == "Selected trial")
        np.testing.assert_allclose(marker.get_offsets()[0], [1, -1])
        page._next_button.click()
        self.assertIn("Convergence reached", page._details_browser.toPlainText())
        self.assertIn("Reduced β = 0.5 · 0.5 = 0.25", page._details_browser.toPlainText())
        self.assertEqual(page._backtracking_table.chain.toPlainText(),
                         "Iteration 2 · β trials: 0.5 → 0.25 (accepted)")
        self.assertEqual([table.item(row, 1).text() for row in range(table.rowCount())], ["0.5", "0.25"])
        page._previous_button.click()
        self.assertEqual(page._iteration_spin.value(), 1)
        self.assertEqual(page._active_trial, frame.chosen_trial)

    def test_halving_backtracking_table_keeps_every_recorded_trial_and_comparison(self):
        page = self.page
        frame = self._run("gradient_halving", "(x-2)**4 + (x-2*y)**2", start="0,0")
        backtracking = page._backtracking_table
        table = backtracking.table
        self.assertEqual(page._result_tabs.tabText(1), "Backtracking")
        self.assertEqual(table.columnCount(), 9)
        self.assertEqual(table.rowCount(), 5)
        self.assertEqual(backtracking.chain.toPlainText(),
                         "Iteration 1 · β trials: 1.0 → 0.5 → 0.25 → 0.125 → 0.0625 (accepted)")
        self.assertEqual(table.verticalScrollBar().value(), 0)
        for index, trial in enumerate(frame.trials):
            with self.subTest(trial=index + 1):
                cells = [table.item(index, column).text() for column in range(9)]
                self.assertEqual(cells[0], str(index + 1))
                self.assertEqual(float(cells[1]), trial.step_size)
                self.assertEqual(cells[2], "(0, 0)")
                self.assertEqual(float(cells[3][1:].partition(",")[0]), frame.direction[0])
                self.assertEqual(cells[4], f"({trial.point[0]:g}, {trial.point[1]:g})")
                self.assertEqual(float(cells[5]), trial.value)
                self.assertEqual(float(cells[6]), frame.current.value)
                self.assertEqual(cells[7:], ["Yes", "Accepted"] if trial.accepted else ["No", "Rejected"])
        page._fit_run_button.click()
        self.assertLess(page._figure.axes[0].get_xlim()[1], frame.trials[0].point[0])
        table.setCurrentCell(0, 1)
        self.assertFalse(page._focus_step.isChecked())
        self.assertGreater(page._figure.axes[0].get_xlim()[1], frame.trials[0].point[0])
        self.assertEqual(table.item(4, 8).background().color().name(), "#193b2a")

    def test_parabola_has_the_actual_curve_support_points_and_interpolant(self):
        page = self.page
        frame = self._run("gradient_parabola", "(x-2)**4/16 + (x-2*y)**2", start="0,0")
        self.assertEqual(page._plot_tabs.tabText(1), "Parabola fit")
        axis = page._method_figure.axes[0]
        lines = {line.get_label(): line for line in axis.lines}
        actual, interpolant = lines["Actual φ(β)"], lines["Interpolating parabola"]
        self.assertGreater(np.max(np.abs(actual.get_ydata() - interpolant.get_ydata())), 1e-6)
        for index, trial in enumerate(frame.trial_points):
            np.testing.assert_allclose(self._trial_markers()[index], [trial.step_size, trial.value])
        page._intermediate_browser.anchorClicked.emit(QUrl("#iteration-2"))
        self.assertEqual(page._iteration_spin.value(), 2)
        page._trial_browser.anchorClicked.emit(QUrl("#trial-2"))
        self.assertIn("Selected row · 2β · f₂", page._details_browser.toPlainText())
        self.assertIn("q′(t)", page._details_browser.toPlainText())
        page._reset_button.click()
        self.assertEqual(page._iteration_spin.value(), 1)

    def test_newton_distinguishes_minimum_indefinite_and_singular_models(self):
        page = self.page
        self._run("newton")
        self.assertEqual(page._plot_tabs.tabText(1), "Quadratic model")
        self.assertTrue(page._initial_step_edit.isHidden())
        labels = page._method_figure.axes[0].get_legend_handles_labels()[1]
        self.assertIn("Newton step · model minimum", labels)
        self.assertIn("β = 1 (full step)", page._details_browser.toPlainText())
        self._run("newton", "x**2 - y**2", start="1,2")
        self.assertIn("indefinite", page._method_figure.axes[0].get_title())
        labels = page._method_figure.axes[0].get_legend_handles_labels()[1]
        self.assertIn("Newton step", labels)
        self.assertNotIn("Newton step · model minimum", labels)
        failed = self._run("newton", "x**2", start="1,2")
        self.assertIsNone(failed.following)
        self.assertIsNone(failed.direction)
        self.assertIn("singular", page._details_browser.toPlainText())
        self.assertIn("Hessian H(xₖ) used", page._details_browser.toPlainText())

    def test_broyden_plot_matrix_update_and_residual_match_the_solver(self):
        page = self.page
        frame = self._run("broyden", "(x-1)**4 + (y+2)**2", start="2,0")
        self.assertEqual(page._plot_tabs.tabText(1), "Broyden update")
        first, second, residual = page._method_figure.axes
        self.assertTrue(first.images and second.images)
        np.testing.assert_allclose(residual.lines[0].get_ydata(), frame.current.gradient)
        np.testing.assert_allclose(residual.lines[1].get_ydata(), frame.following.gradient)
        self.assertIn("Updated inverse Bₖ₊₁", page._details_browser.toPlainText())
        page._iteration_spin.setValue(page._iteration_spin.maximum())
        self.assertFalse(page._method_figure.axes[1].images)
        self.assertIn("Stopped at the gradient tolerance", page._details_browser.toPlainText())

    def test_manual_view_and_convergence_plot_follow_the_selected_iteration(self):
        page = self.page
        self._run("gradient_halving")
        page._fit_run_button.click()
        self.assertFalse(page._focus_step.isChecked())
        axis = page._figure.axes[0]
        axis.set_xlim(-.5, 1.9)
        page._refresh_trajectory_view()
        self.assertEqual(page._figure.axes[0].get_xlim(), (-.5, 1.9))
        page._next_button.click()
        self.assertEqual(page._figure.axes[0].get_xlim(), (-.5, 1.9))
        norms = page._convergence_canvas.figure.axes[1]
        np.testing.assert_allclose(norms.lines[0].get_ydata(), [step.gradient_norm for step in page._result.steps])
        self.assertIn("ε = 1e-06", norms.get_legend_handles_labels()[1])
        page._focus_step.setChecked(True)
        self.assertNotEqual(page._figure.axes[0].get_xlim(), (-.5, 1.9))

    def test_invalid_trials_are_visible_as_markers_at_their_beta(self):
        self._run("gradient_halving", "log(x)", "x", "1")
        frame = self.page._frames[0]
        self.assertIsNone(frame.trials[0].value)
        self.assertEqual(self._trial_markers()[0][0], 1)
        table = self.page._backtracking_table.table
        self.assertEqual(table.item(0, 5).text(), "—")
        self.assertEqual(table.item(0, 7).text(), "No")
        self.assertEqual(table.item(0, 8).text(), "Rejected")
        table.setCurrentCell(0, 1)
        self.assertIn("Invalid function value", self.page._details_browser.toPlainText())

    def test_three_dimensional_trajectory_uses_the_current_fixed_coordinate(self):
        self._run("newton", "(x-1)**4 + (y+2)**2 + (z-3)**4", "x,y,z", "2,0,4")
        with patch.object(ParsedFunction, "evaluate_grid", autospec=True,
                          side_effect=ParsedFunction.evaluate_grid) as evaluate_grid:
            self.page._next_button.click()
        current = self.page._frames[1].current.point
        self.assertNotEqual(current[2], 4.)
        self.assertAlmostEqual(evaluate_grid.call_args.args[-1], current[2])
        self.assertIn("higher coordinates fixed", self.page._figure.axes[0].get_title())
        self.assertIn("full Newton direction", self.page._method_figure.axes[0].get_xlabel())
