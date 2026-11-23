import os
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from topics.integer_programming.page import IntegerProgrammingPage


class HullViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = IntegerProgrammingPage()
        self.page.resize(1120, 700)
        self.page.show()
        self._choose(self.page._mode_combo, "hull")
        self.controls = self.page._hull_controls
        self.workspace = self.page._hull_workspace
        self.application.processEvents()

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.application.processEvents()

    @staticmethod
    def _choose(combo, data):
        combo.setCurrentIndex(combo.findData(data))

    def _labels(self):
        axis = self.workspace._canvas.figure.axes[0]
        return set(axis.get_legend_handles_labels()[1])

    def test_point_mode_constructs_the_hull_and_preserves_manual_axes(self):
        self.assertTrue(self.page._editor_fields.isHidden())
        self.controls.points.setPlainText("0, 0\n1, 1\n2, 0\n1, 1\n1, 1/2")
        self.page._convert_button.click()
        self.assertNotIn("Convex hull", self._labels())
        self.workspace._next.click()
        self.assertIn("Consider", self.workspace.details.toPlainText())
        self.workspace._final.click()
        self.assertIn("Hull vertices", self._labels())
        self.assertIn("Other input points", self._labels())
        self.assertIn("a polygon", self.workspace.details.toPlainText())
        minimum, maximum = self.controls.axis_edits[0]
        minimum.setText("-2")
        maximum.setText("4")
        maximum.editingFinished.emit()
        axis = self.workspace._canvas.figure.axes[0]
        self.assertEqual(axis.get_xlim(), (-2, 4))
        self.assertEqual(axis.get_aspect(), 1.0)
        self.workspace._hull_switch.click()
        self.assertNotIn("Convex hull", self._labels())
        self.assertEqual(axis.get_xlim(), (-2, 4))
        self.workspace._reset.click()
        self.assertEqual(self.workspace._index, 0)

    def test_integer_mode_loads_lecture_and_updates_plot_after_model_edits(self):
        self._choose(self.controls.method, "integer")
        self.page._load_hull_example(self.controls.examples.currentText())
        self.assertTrue(self.page._editor_fields.isVisible())
        self.assertIn("Integer hull", self._labels())
        self.assertIn("LP optimum", self._labels())
        self.assertIn("ILP optimum", self._labels())
        self.assertIn("= 1", self.workspace.details.toPlainText())
        self.page._right_hand_side_edits[1].setText("8")
        self.page._convert_button.click()
        self.assertEqual(self.workspace._integer.lp.optimum.value, 4)
        self.workspace._lp_switch.click()
        self.assertNotIn("LP region P", self._labels())
        self.assertIn("Integer hull", self._labels())
        self.controls.max_points.setValue(1)
        self.page._convert_button.click()
        self.assertNotIn("Integer hull", self._labels())
        self.assertIn("Point limit reached", self.page._status_label.text())

    def test_cuts_plot_and_comparison_follow_playback(self):
        self._choose(self.controls.method, "cuts")
        self.page._load_hull_example(self.controls.examples.currentText())
        self.assertNotIn("Current LP optimum", self._labels())
        self.assertNotIn("Objective comparison", self.workspace.details.toPlainText())
        self.workspace._next.click()
        self.assertIn("Current LP optimum", self._labels())
        self.assertIn("Upper bound", self.workspace.details.toPlainText())
        self.workspace._next.click()
        self.assertTrue(self.controls.add_cut.isEnabled())
        self.controls.manual_cut.setText("0, 1, 1")
        self.controls.add_cut.click()
        self.assertIn("Invalid cut", self.controls.cut_message.text())
        self.workspace._next.click()
        self.assertIn("Proposed cut", self._labels())
        self.assertIn("Removed area", self._labels())
        self.assertIn("3/4 · C1 + 1/4 · C2", self.workspace.details.toPlainText())
        self.workspace._next.click()
        self.assertIn("Current LP region", self._labels())
        self.assertNotIn("Current LP optimum", self._labels())
        self.assertNotIn("Removed area", self._labels())
        self.workspace._next.click()
        self.assertIn("8/3", self.workspace.details.toPlainText())
        self.workspace._previous.click()
        self.assertNotIn("8/3", self.workspace.details.toPlainText())
        self.workspace._reset.click()
        self.assertEqual(self.workspace._cuts.index, 0)
        self.assertNotIn("Current LP optimum", self._labels())

    def test_switching_to_legacy_modes_restores_the_shared_editor(self):
        self.assertGreaterEqual(self.page._status_row.indexOf(self.page._convert_button), 0)
        self._choose(self.page._mode_combo, "lp")
        self.assertIs(self.page._standard_top.widget(0), self.page._input_scroll)
        self.assertGreaterEqual(self.page._input_button_layout.indexOf(self.page._convert_button), 0)
        self.assertTrue(self.page._editor_fields.isVisible())
        self.assertIsNotNone(self.page._lp_result)
        self._choose(self.page._mode_combo, "bb")
        self.assertIsNotNone(self.page._run)
        self._choose(self.page._mode_combo, "hull")
        self.assertIs(self.workspace.widget(0), self.page._input_scroll)
