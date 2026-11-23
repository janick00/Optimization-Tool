"""Check numerical steps against the optimization formula sheet."""

import unittest

import numpy as np

from topics.unconstrained_optimization.methods import (
    aitken_step,
    broyden_step,
    broyden_update,
    newton_step,
    parabola_fitting,
    successive_halving,
)
from topics.unconstrained_optimization.objective import parse_function


class OptimizationMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.function = parse_function("(x**2 - 2*x*y + x)**2", "x, y")

    def test_parabola_fitting_matches_formula_sheet(self) -> None:
        result = parabola_fitting(self.function.evaluate, [2, 2], [-4, 16], 1 / 32)

        self.assertEqual([trial.step_size for trial in result.trials[:2]], [1 / 32, 1 / 16])
        self.assertIsNotNone(result.selected)
        self.assertAlmostEqual(result.selected.step_size, 20099 / 771296)
        np.testing.assert_allclose(result.selected.point, [405747 / 192824, 76313 / 48206])

    def test_parabola_fitting_can_use_a_minimum_beyond_the_samples(self) -> None:
        def objective(x):
            return x[0] ** 2

        result = parabola_fitting(objective, [1], [2], 1 / 8)
        self.assertEqual(result.selected.step_size, 0.5)

    def test_parabola_fitting_prefers_original_beta_on_a_tie(self) -> None:
        calls = []

        def objective(point):
            calls.append(tuple(point))
            return point[0] ** 2

        result = parabola_fitting(objective, [1], [2], 0.5)

        self.assertIs(result.selected, result.trials[0])
        self.assertEqual(result.fitted.value, result.selected.value)
        self.assertEqual(calls, [(1,), (0,), (-1,)])

    def test_invalid_curvature_falls_back_only_when_beta_improves(self) -> None:
        cases = (
            (lambda x: x[0], [1], [1], "numerically zero", True),
            (lambda x: -(x[0] ** 2), [1], [-2], "negative", True),
            (lambda x: x[0], [0], [-1], "numerically zero", False),
        )
        for objective, point, gradient, reason, improves in cases:
            with self.subTest(reason=reason, improves=improves):
                result = parabola_fitting(objective, point, gradient)
                self.assertIsNone(result.fitted)
                self.assertIn(reason, result.fit_issue)
                self.assertEqual(result.selected is not None, improves)
                if improves:
                    self.assertIs(result.selected, result.trials[0])
                else:
                    self.assertIn("No improving step", result.failure_reason)

    def test_parabola_fitting_rejects_zero_negative_and_nonfinite_estimates(self) -> None:
        for linear_term in (0.0, 1.0):
            with self.subTest(linear_term=linear_term):
                result = parabola_fitting(
                    lambda x: x[0] ** 2 + linear_term * x[0], [0], [-1]
                )
                self.assertIsNone(result.fitted)
                self.assertIsNone(result.selected)
                self.assertLessEqual(result.beta_star, 0)
                self.assertIn("positive", result.fit_issue)

        result = parabola_fitting(
            lambda x: {-2.0: -5e307, -1.0: 0.0, 0.0: 1e308}[float(x[0])],
            [0], [1], 1,
        )
        self.assertIsNone(result.fitted)
        self.assertTrue(np.isposinf(result.beta_star))
        self.assertIs(result.selected, result.trials[0])

    def test_parabola_fitting_handles_invalid_samples_and_fitted_evaluations(self) -> None:
        def objective(x):
            return {0.0: 4.0, 1.0: 0.0, 2.0: 2.0}.get(x[0], float("inf"))

        result = parabola_fitting(objective, [0], [-1], 1)
        self.assertIsNone(result.fitted.value)
        self.assertIs(result.selected, result.trials[0])
        self.assertIn("Invalid function value", result.fit_issue)

        result = parabola_fitting(lambda x: x[0] if x[0] >= 0 else np.nan, [1], [1], 0.75)
        self.assertIsNone(result.samples[2].value)
        self.assertIsNone(result.interpolation)
        self.assertIs(result.selected, result.trials[0])

    def test_parabola_fitting_does_not_select_two_beta_as_a_third_candidate(self) -> None:
        result = parabola_fitting(
            lambda x: {0.0: 1, 1.0: 1, 2.0: 0}[x[0]], [0], [-1], 1
        )
        self.assertTrue(result.samples[2].accepted)
        self.assertIsNone(result.selected)

    def test_successive_halving_stops_at_first_improvement(self) -> None:
        result = successive_halving(self.function.evaluate, [2, 2], [-4, 16], 1 / 16)

        self.assertEqual([trial.step_size for trial in result.trials], [1 / 16, 1 / 32])
        self.assertEqual(result.selected.point, (2.125, 1.5))
        self.assertAlmostEqual(result.selected.value, 0.070556640625)

    def test_newton_and_broyden_match_formula_sheet_iterates(self) -> None:
        point = np.array([2.0, 2.0])
        gradient = self.function.gradient_at(point)
        hessian = self.function.hessian_at(point)
        np.testing.assert_allclose(point + newton_step(gradient, hessian), [4 / 3, 3 / 2])

        inverse = np.linalg.inv(hessian)
        displacement = broyden_step(gradient, inverse)
        next_point = point + displacement
        np.testing.assert_allclose(next_point, [4 / 3, 3 / 2])

        next_gradient = self.function.gradient_at(next_point)
        inverse = broyden_update(inverse, displacement, next_gradient - gradient)
        np.testing.assert_allclose(
            next_point + broyden_step(next_gradient, inverse), [20 / 19, 49 / 38]
        )

    def test_aitken_reports_both_iterates_and_the_residual(self) -> None:
        def fixed_point(x):
            return 0.5 * x + np.array([1, -2])

        step = aitken_step(fixed_point, [0, 0])

        self.assertEqual(step.first, (1, -2))
        self.assertEqual(step.second, (1.5, -3))
        self.assertEqual(step.accelerated, (2, -4))
        self.assertEqual(step.residual, 0.0)

    def test_aitken_and_broyden_reject_zero_denominators(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            aitken_step(lambda x: x, [1])
        with self.assertRaises(ZeroDivisionError):
            broyden_update([[1]], [1], [0])
