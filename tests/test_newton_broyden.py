"""Validate Newton and inverse Broyden against analytical examples."""

import unittest

import numpy as np

from topics.unconstrained_optimization.formatting import (
    format_intermediate_details_html,
    format_optimization_details_html,
)
from topics.unconstrained_optimization.methods import broyden_step, broyden_update
from topics.unconstrained_optimization.models import OptimizationMethod, OptimizationSettings
from topics.unconstrained_optimization.objective import parse_function
from topics.unconstrained_optimization.solver import optimize


METHODS = (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN)


class NewtonBroydenTests(unittest.TestCase):
    def test_supplied_quadratic_reaches_analytical_minimum_in_one_step(self) -> None:
        function = parse_function("x**2 + 2*y**2 - 2*x - 4*y", "x,y")
        for method in METHODS:
            with self.subTest(method=method):
                result = optimize(function, method, (4, 3))
                self.assertTrue(result.converged)
                self.assertIn("strict local minimum", result.message)
                self.assertEqual(len(result.steps), 2)
                self.assertEqual(result.steps[0].gradient, (6, 8))
                self.assertEqual(result.final_point, (1, 1))
                self.assertEqual(result.final_value, -3)
                self.assertEqual(result.steps[1].gradient_norm, 0)
                details = result.steps[1].second_order
                if method is OptimizationMethod.NEWTON:
                    self.assertEqual(details.matrix, ((2, 0), (0, 4)))
                else:
                    self.assertEqual(details.matrix, ((0.5, 0), (0, 0.25)))
                    self.assertEqual(details.displacement, (-3, -2))
                    self.assertEqual(details.gradient_change, (-6, -8))

    def test_newton_uses_current_hessian_and_broyden_reuses_its_update_on_quartic(self) -> None:
        function = parse_function("(x-1)**4 + (y+2)**2", "x,y")
        for method in METHODS:
            with self.subTest(method=method):
                result = optimize(
                    function, method, (2, 0),
                    OptimizationSettings(tolerance=1e-10, max_iterations=100),
                )
                self.assertTrue(result.converged, result.message)
                self.assertGreater(len(result.steps), 3)
                self.assertLess(result.final_value, 1e-12)
                np.testing.assert_allclose(result.final_point, [1, -2], atol=4e-4)

                expected_inverse = np.diag([1 / 12, 0.5])
                for current, following in zip(result.steps, result.steps[1:]):
                    x, y = current.point
                    dx, dy = x - 1, y + 2
                    gradient = np.array([4 * dx**3, 2 * dy])
                    np.testing.assert_allclose(current.gradient, gradient, atol=1e-15)
                    details = following.second_order
                    if method is OptimizationMethod.NEWTON:
                        np.testing.assert_allclose(details.matrix, np.diag([12 * dx**2, 2]))
                        expected_point = [1 + (2 / 3) ** following.iteration, -2]
                    else:
                        np.testing.assert_allclose(details.matrix, expected_inverse, rtol=1e-9)
                        expected_point = np.asarray(current.point) - expected_inverse @ gradient
                        d = np.asarray(following.point) - current.point
                        g = np.asarray(following.gradient) - current.gradient
                        np.testing.assert_allclose(details.displacement, d)
                        np.testing.assert_allclose(details.gradient_change, g)
                        if following is not result.steps[-1]:
                            row = d @ expected_inverse
                            expected_inverse -= np.outer(expected_inverse @ g - d, row) / (row @ g)
                            np.testing.assert_allclose(expected_inverse @ g, d, atol=1e-14)
                    np.testing.assert_allclose(following.point, expected_point, atol=1e-11)
                    self.assertAlmostEqual(following.value, function.evaluate(following.point))
                    self.assertAlmostEqual(following.gradient_norm, np.linalg.norm(following.gradient))

    def test_three_variables_and_mixed_partials(self) -> None:
        function = parse_function(
            "(x-1)**2 + 2*(y+2)**2 + 3*(z-3)**2 + (x-1)*(z-3)", "x,y,z"
        )
        point = (4, 1, 5)
        np.testing.assert_array_equal(function.gradient_at(point), [8, 12, 15])
        np.testing.assert_array_equal(function.hessian_at(point), [[2, 0, 1], [0, 4, 0], [1, 0, 6]])
        for method in METHODS:
            with self.subTest(method=method):
                result = optimize(function, method, point)
                self.assertTrue(result.converged)
                np.testing.assert_allclose(result.final_point, [1, -2, 3])

    def test_stationary_start_classifies_saddle_and_degenerate_hessian(self) -> None:
        cases = (
            ("x**2 - y**2", "x,y", (0, 0), "saddle point"),
            ("x**4", "x", (0,), "inconclusive"),
            ("x**(4/3)", "x", (0,), "classification unavailable"),
        )
        for method in METHODS:
            for expression, variables, point, classification in cases:
                with self.subTest(method=method, expression=expression):
                    result = optimize(parse_function(expression, variables), method, point)
                    self.assertTrue(result.converged)
                    self.assertEqual(len(result.steps), 1)
                    self.assertIn(classification, result.message)

    def test_newton_preserves_uphill_full_step_and_labels_maximum(self) -> None:
        result = optimize(parse_function("-x**2", "x"), OptimizationMethod.NEWTON, (1,))
        self.assertTrue(result.converged)
        self.assertEqual(result.final_point, (0,))
        self.assertGreater(result.final_value, result.steps[0].value)
        self.assertIn("increased", result.steps[1].note)
        self.assertIn("strict local maximum", result.message)
        self.assertIn("increased", format_intermediate_details_html(result))
        self.assertIn("strict local maximum", format_optimization_details_html(result))

    def test_bad_hessians_and_stagnation_stop_without_duplicate_points(self) -> None:
        cases = (
            ("x**2", "x,y", (1, 0), "singular"),
            ("x**2 + 1e-14*y**2", "x,y", (1, 1), "ill-conditioned"),
            ("x + x**(4/3)", "x", (0,), "Hessian is not finite"),
            ("x + cos(10**10*(x-1))", "x", (1,), "did not change"),
            ("log(x)-x", "x", (3,), "Function is not finite"),
        )
        for method in METHODS:
            for expression, variables, point, reason in cases:
                with self.subTest(method=method, expression=expression):
                    result = optimize(parse_function(expression, variables), method, point)
                    self.assertFalse(result.converged)
                    self.assertEqual(len(result.steps), 1)
                    self.assertIn(reason, result.message)

    def test_iteration_limit_and_strict_tolerance(self) -> None:
        quartic = parse_function("(x-1)**4", "x")
        quadratic = parse_function("x**2", "x")
        for method in METHODS:
            with self.subTest(method=method):
                limited = optimize(quartic, method, (2,), OptimizationSettings(max_iterations=1))
                self.assertFalse(limited.converged)
                self.assertEqual(len(limited.steps), 2)
                self.assertIn("Maximum number", limited.message)

                equal = optimize(quadratic, method, (1,), OptimizationSettings(tolerance=2))
                self.assertTrue(equal.converged)
                self.assertEqual(len(equal.steps), 2)

    def test_nonfinite_initial_objective_is_reported_as_an_input_error(self) -> None:
        function = parse_function("log(x)", "x")
        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "Function is not finite"):
                    optimize(function, method, (-1,))

    def test_broyden_checks_convergence_before_trying_an_update(self) -> None:
        result = optimize(
            parse_function("x**2", "x"), OptimizationMethod.BROYDEN, (1e-8,),
            OptimizationSettings(tolerance=1e-10),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.final_point, (0,))
        self.assertEqual(result.steps[-1].gradient_norm, 0)

    def test_broyden_reports_bad_denominator_only_if_another_step_is_needed(self) -> None:
        function = parse_function("x**3/3 + x**2/2 + x", "x")
        result = optimize(function, OptimizationMethod.BROYDEN, (0,))
        self.assertFalse(result.converged)
        self.assertEqual(result.final_point, (-1,))
        self.assertIn("denominator", result.message)
        self.assertEqual(result.steps[1].second_order.gradient_change, (0,))

        capped = optimize(function, OptimizationMethod.BROYDEN, (0,), OptimizationSettings(max_iterations=1))
        self.assertFalse(capped.converged)
        self.assertEqual(capped.final_point, (-1,))
        self.assertIn("Maximum number", capped.message)

    def test_broyden_denominator_is_scale_aware_and_rejects_cancellation(self) -> None:
        np.testing.assert_allclose(broyden_update([[1]], [1e-180], [2e-180]), [[0.5]])
        with self.assertRaisesRegex(ZeroDivisionError, "denominator"):
            broyden_update(np.eye(2), [1, 1], [1, -1 + 1e-15])
        with self.assertRaisesRegex(ArithmeticError, "not finite"):
            broyden_step([2], [[1e308]])
        with self.assertRaisesRegex(ArithmeticError, "non-finite"):
            broyden_update([[1]], [1], [np.nan])
