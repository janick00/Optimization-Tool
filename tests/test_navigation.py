"""GUI tests for initial topic selection and navigation."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on the local environment
    QApplication = None  # type: ignore[assignment]

from gui.main_window import MainWindow
from gui.topic_registry import TOPIC_REGISTRY


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
        for topic in TOPIC_REGISTRY:
            self.window.show_topic(topic.identifier)
            self.assertEqual(self.window.current_topic_id, topic.identifier)
            self.assertTrue(self.window._buttons_by_topic_id[topic.identifier].isChecked())

    def test_problem_formulation_rebuilds_fields_from_selected_counts(self) -> None:
        page = self.window._page_stack.currentWidget()
        page._variable_count_combo.setCurrentIndex(2)  # three variables
        page._constraint_count_combo.setCurrentIndex(0)  # one constraint
        self.application.processEvents()

        self.assertEqual(len(page._objective_edits), 3)
        self.assertEqual(len(page._constraint_coefficient_edits), 1)
        self.assertEqual(len(page._domain_combos), 3)
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
        self.assertEqual(str(page._last_function.expression), "x1**2 + x2**2")
        self.assertEqual(page._initial_step_edit.text(), "0.25")
        self.assertIn("Successive halving candidates", page._intermediate_browser.toPlainText())
        self.assertIn("done!", page._intermediate_browser.toPlainText())
        self.assertIn("Function and derivatives", page._details_browser.toPlainText())
        self.assertIn("Iteration results", page._details_browser.toPlainText())

