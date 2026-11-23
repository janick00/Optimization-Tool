import os
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from topics.integer_programming.page import IntegerProgrammingPage


class ILPLessonViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = IntegerProgrammingPage()
        self.page.resize(1120, 670)
        self.page.show()
        self.addCleanup(self._close_page)

    def _close_page(self):
        self.page.close()
        self.page.deleteLater()
        self.application.processEvents()

    @staticmethod
    def _choose(combo, mode):
        combo.setCurrentIndex(combo.findData(mode))

    def _lesson(self, mode):
        self._choose(self.page._mode_combo, mode)
        self.application.processEvents()
        return self.page._lesson_pages[mode]

    def test_knapsack_greedy_playback_and_capacity_diagram(self):
        view = self._lesson("knapsack")
        self.assertEqual(view.items.item(1, 3).text(), "3/2")
        view.solve.click()
        self.assertIn("Input order: x1", view.details.toPlainText())
        self.assertNotIn("65/2", view.details.toPlainText())
        view.playback.go(5)
        axis = view.canvas.figure.axes[0]
        self.assertTrue(any(patch.get_hatch() == "///" for patch in axis.patches))
        self.assertIn("Pack x5 = 1/6", view.details.toPlainText())
        view.playback.next.click()
        self.assertIn("(1, 1, 1, 1/6, 0)", view.details.toPlainText())
        self.assertIn("(1, 1, 1, 0, 1/6)", view.details.toPlainText())
        view.playback.next.click()
        self.assertIn("LB = 31", view.details.toPlainText())
        self.assertFalse(any(patch.get_hatch() == "///" for patch in view.canvas.figure.axes[0].patches))
        view.items.item(0, 1).setText("0")
        self.assertEqual(view.items.item(0, 3).text(), "—")
        view.solve.click()
        self.assertIn("positive weight", view.status.text())
        self.assertEqual(view.playback.length, 0)

    def test_knapsack_tree_is_reversible_and_auto_stops_on_navigation(self):
        view = self._lesson("knapsack")
        self._choose(view.method, "bb")
        view.solve.click()
        self.assertEqual(set(view._nodes), {1})
        self.assertNotIn("Optimal binary", view.details.toPlainText())
        view.playback.auto.click()
        view.playback.timer.timeout.emit()
        self.assertEqual(view.playback.index, 1)
        self.assertIn("UB = 65/2", view.details.toPlainText())
        view.playback.go(3)
        self.assertEqual(set(view._nodes), {1, 2, 3})
        self.assertIn("x5 = 1", view.tree._node_items[2].toolTip())
        view.playback.previous.click()
        self.assertEqual(set(view._nodes), {1})
        view.playback.go(view.playback.length - 1)
        self.assertIn("Optimal binary packing", view.details.toPlainText())
        view.playback.reset.click()
        view.playback.auto.click()
        self._lesson("spanning")
        self.assertFalse(view.playback.timer.isActive())
        self._choose(self.page._mode_combo, "hull")
        self.assertIs(self.page._main_stack.currentWidget(), self.page._hull_workspace)
        self._choose(self.page._mode_combo, "bb")
        self.assertIsNotNone(self.page._run)

    def test_spanning_stepwise_comparison_3d_and_tree(self):
        view = self._lesson("spanning")
        view.solve.click()
        self.assertIsNone(view._weak)
        self.assertTrue(view.diagram.isHidden())
        for _ in range(5):
            view.playback.next.click()
        self.assertIn("3/2 < 2", view.details.toPlainText())
        self.assertIn("excludes the weak LP point", view.details.toPlainText())
        self.assertIn("Group", view.graph._nodes[0].toolTip())
        view.diagram.click()
        view.canvas.draw()
        self.assertEqual(len(view.canvas.figure.axes), 1)
        view.playback.next.click()
        view.canvas.draw()
        self.assertEqual(len(view.canvas.figure.axes), 2)
        view.playback.next.click()
        self.assertIn("8 ≤ 9", view.details.toPlainText())
        view.playback.next.click()
        self.assertIn("total weight 9", view.details.toPlainText())
        self.assertIs(view.views.currentWidget(), view.graph)
        view.playback.reset.click()
        self.assertNotIn("total weight", view.details.toPlainText())
        self.assertTrue(view.diagram.isHidden())

    def test_spanning_graph_editing_remaps_endpoints_and_validates_edges(self):
        view = self._lesson("spanning")
        view.nodes.item(1, 0).setText("B")
        self.assertEqual(view._graph.names[1], "B")
        view.nodes.setCurrentCell(1, 0)
        view._remove_node()
        self.assertEqual(view._graph.names, ("1", "3"))
        self.assertEqual(view._graph.edges(), ((0, 1),))
        self.assertEqual(view._graph.weight((0, 1)), 7)
        view._append_edge(("2", "1", "6"))
        view._refresh_graph()
        self.assertIsNone(view._graph)
        self.assertIn("duplicates", view.status.text())
