"""Tests for Aitken estimates along an ordinary scalar fixed-point sequence."""

import math
import unittest

import numpy as np
from matplotlib.figure import Figure

from topics.unconstrained_optimization.aitken_visualization import (
    plot_aitken_residuals,
    plot_aitken_values,
)
from topics.unconstrained_optimization.fixed_point_input import parse_fixed_point_function
from topics.unconstrained_optimization.methods import iterate_fixed_point


class AitkenTests(unittest.TestCase):
    def test_cosine_values_and_all_requested_counts(self) -> None:
        for count in (3, 5, 10, 20):
            with self.subTest(count=count):
                result = iterate_fixed_point(parse_fixed_point_function("cos(x)"), 1, count)
                expected_points = [1.0]
                for _ in range(count - 1):
                    expected_points.append(math.cos(expected_points[-1]))
                self.assertEqual(result.points, tuple(expected_points))
                self.assertEqual([row.index for row in result.estimates], list(range(2, count)))
                self.assertIsNone(result.stop_reason)
                self.assertAlmostEqual(result.points[1], 0.540302306, places=9)
                self.assertAlmostEqual(result.points[2], 0.857553216, places=9)
                self.assertAlmostEqual(result.estimates[0].estimate, 0.728010361, places=9)
                np.testing.assert_allclose(
                    result.residuals, [abs(math.cos(x) - x) for x in expected_points],
                    atol=0, rtol=0,
                )
                for row in result.estimates:
                    older, previous, current = expected_points[row.index - 2:row.index + 1]
                    denominator = current - 2 * previous + older
                    estimate = current - (current - previous)**2 / denominator
                    self.assertAlmostEqual(row.denominator, denominator, places=14)
                    self.assertAlmostEqual(row.estimate, estimate, places=14)
                    self.assertAlmostEqual(row.residual, abs(math.cos(estimate) - estimate), places=14)
                if count == 20:
                    self.assertAlmostEqual(result.estimates[-1].estimate, 0.739085133, places=7)

    def test_fixed_point_and_numerically_zero_denominators_keep_all_points(self) -> None:
        cases = ("x", "x + 1", "x + 1 + 1e-16 * x")
        for expression in cases:
            with self.subTest(expression=expression):
                result = iterate_fixed_point(parse_fixed_point_function(expression), 1, 10)
                self.assertEqual(len(result.points), 10)
                self.assertIsNone(result.stop_reason)
                for row in result.estimates:
                    self.assertIsNone(row.estimate)
                    self.assertIsNone(row.residual)
                    self.assertIn("numerically zero", row.issue)
        self.assertEqual(iterate_fixed_point(lambda x: x, 1, 5).residuals, (0.0,) * 5)

    def test_residual_failure_at_estimate_does_not_affect_ordinary_sequence(self) -> None:
        def map_with_hole(x):
            if x == 0:
                raise ValueError("outside the function domain")
            return 2 * x

        result = iterate_fixed_point(map_with_hole, 1, 5)
        self.assertEqual(result.points, (1, 2, 4, 8, 16))
        self.assertEqual(result.residuals, (1, 2, 4, 8, 16))
        self.assertIsNone(result.stop_reason)
        for row in result.estimates:
            self.assertEqual(row.estimate, 0)
            self.assertIsNone(row.residual)
            self.assertIn("Residual unavailable", row.issue)

    def test_invalid_ordinary_evaluations_retain_valid_prefix(self) -> None:
        map_with_limit = parse_fixed_point_function("sqrt(x) - 2")
        result = iterate_fixed_point(map_with_limit, 4, 10)
        self.assertEqual(result.points, (4, 0, -2))
        self.assertEqual(result.residuals, (4, 2, None))
        self.assertIn("x[3]", result.stop_reason)
        self.assertEqual(len(result.estimates), 1)

        for function in (lambda _: math.nan, lambda _: math.inf, parse_fixed_point_function("exp(x)")):
            with self.subTest(function=function):
                bad = iterate_fixed_point(function, 1000)
                self.assertEqual(bad.points, (1000,))
                self.assertIn("x[1]", bad.stop_reason)

    def test_final_residual_probe_does_not_append_or_invalidate_an_iterate(self) -> None:
        result = iterate_fixed_point(parse_fixed_point_function("sqrt(x) - 2"), 4, 3)
        self.assertEqual(result.points, (4, 0, -2))
        self.assertEqual(result.residuals, (4, 2, None))
        self.assertIsNone(result.stop_reason)
        self.assertIn("final ordinary residual", result.last_residual_issue)

    def test_overflow_in_differences_does_not_stop_finite_ordinary_values(self) -> None:
        result = iterate_fixed_point(lambda x: -x, 1e308, 5)
        self.assertEqual(result.points, (1e308, -1e308, 1e308, -1e308, 1e308))
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.residuals, (None,) * 5)
        self.assertTrue(all(row.estimate is None for row in result.estimates))
        self.assertTrue(all("not finite" in row.issue for row in result.estimates))

    def test_expression_support_and_input_validation(self) -> None:
        function = parse_fixed_point_function("sin(x) + exp(-x) + x^2 + floor(x)")
        self.assertAlmostEqual(function(2.0), math.sin(2) + math.exp(-2) + 6)
        for expression in ("", "y + 1", "foo(x)", "x.real", "[x]"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    parse_fixed_point_function(expression)
        with self.assertRaisesRegex(ValueError, "three|3"):
            iterate_fixed_point(math.cos, 1, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            iterate_fixed_point(math.cos, math.nan)

    def test_plots_use_recorded_values_and_switch_scale_when_residuals_include_zero(self) -> None:
        result = iterate_fixed_point(math.cos, 1)
        figure = Figure()
        plot_aitken_values(figure, result)
        ordinary, estimates = figure.axes[0].lines
        np.testing.assert_array_equal(ordinary.get_ydata(), result.points)
        np.testing.assert_array_equal(estimates.get_ydata(), [row.estimate for row in result.estimates])
        plot_aitken_residuals(figure, result)
        ordinary, estimates = figure.axes[0].lines
        np.testing.assert_array_equal(ordinary.get_ydata(), result.residuals)
        np.testing.assert_array_equal(estimates.get_ydata(), [row.residual for row in result.estimates])
        self.assertEqual(figure.axes[0].get_yscale(), "log")

        result = iterate_fixed_point(lambda x: x, 1, 3)
        plot_aitken_residuals(figure, result)
        self.assertEqual(figure.axes[0].get_yscale(), "linear")
        self.assertTrue(np.isnan(figure.axes[0].lines[1].get_ydata()[0]))
