"""Tests for unconstrained continuous optimization methods."""

import unittest
from unittest.mock import patch

import numpy as np

from topics.unconstrained_optimization.formatting import (
    format_intermediate_details_html,
    format_optimization_details_html,
)
from topics.unconstrained_optimization.models import OptimizationMethod, OptimizationSettings
from topics.unconstrained_optimization.objective import ParsedFunction, parse_function
from topics.unconstrained_optimization.solver import optimize


class UnconstrainedOptimizationTests(unittest.TestCase):
    def test_parser_creates_gradient_and_hessian(self) -> None:
        function = parse_function("x1**2 + 2*x1*x2 + x2**2", "x1, x2")
        self.assertEqual(function.dimension, 2)
        self.assertEqual(tuple(map(str, function.gradient_expression)), ("2*x1 + 2*x2", "2*x1 + 2*x2"))
        self.assertEqual(function.gradient_at((1.0, 2.0)).tolist(), [6.0, 6.0])
        self.assertEqual(function.hessian_at((1.0, 2.0)).tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_grid_evaluation_preserves_shape_and_undefined_values(self) -> None:
        x = np.array([[-1.0, 0.0, 4.0]])
        y = np.array([[1.0], [2.0]])
        function = parse_function("sqrt(x) + y + z", "x, y, z")
        values = function.evaluate_grid(x, y, 3.0)

        np.testing.assert_allclose(values, [[np.nan, 4.0, 6.0], [np.nan, 5.0, 7.0]])
        constant = parse_function("5", "x, y")
        np.testing.assert_array_equal(constant.evaluate_grid(x, y), np.full((2, 3), 5.0))

    def test_grid_evaluation_supports_expressions_that_require_scalar_sampling(self) -> None:
        function = parse_function("Min(x, 0)", "x")
        values = function.evaluate_grid(np.array([-2.0, 0.0, 2.0]))

        np.testing.assert_array_equal(values, [-2.0, 0.0, 0.0])

    def test_iterative_methods_reach_quadratic_bowl_minimum(self) -> None:
        function = parse_function("x1**2 + x2**2", "x1, x2")
        settings = OptimizationSettings(tolerance=1e-8, max_iterations=20)
        for method in OptimizationMethod:
            if method is OptimizationMethod.GRADIENT_HALVING:
                continue
            with self.subTest(method=method):
                result = optimize(function, method, (3.0, 1.0), settings)
                self.assertTrue(result.converged)
                self.assertAlmostEqual(result.final_value, 0.0, places=8)
                self.assertGreaterEqual(len(result.steps), 2)

    def test_gradient_descent_records_line_search_trials(self) -> None:
        function = parse_function("(x1 - 2)**4 + (x1 - 2*x2)**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_PARABOLA,
            (0.0, 0.0),
            OptimizationSettings(max_iterations=3),
        )
        self.assertTrue(result.parabola_iterations[0][1].trials)

    def test_successive_halving_starts_at_beta_and_only_halves(self) -> None:
        function = parse_function("(x1 - 2)**4 + (x1 - 2*x2)**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_HALVING,
            (0.0, 0.0),
            OptimizationSettings(initial_step=1.0, max_iterations=1),
        )

        first_iteration = result.steps[1]
        self.assertAlmostEqual(first_iteration.step_size, 1 / 16)
        self.assertEqual(
            [trial.step_size for trial in first_iteration.trials],
            [1.0, 0.5, 0.25, 0.125, 0.0625],
        )
        self.assertEqual(
            [trial.phase for trial in first_iteration.trials],
            ["initial beta", "halving", "halving", "halving", "halving"],
        )
        self.assertIn("Step accepted", first_iteration.note)

    def test_user_defined_beta_is_the_first_line_search_trial(self) -> None:
        function = parse_function("x1**2 + x2**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_HALVING,
            (3.0, 1.0),
            OptimizationSettings(initial_step=0.25, max_iterations=1),
        )

        first_trial = result.steps[1].trials[0]
        self.assertAlmostEqual(first_trial.step_size, 0.25)
        self.assertEqual(first_trial.phase, "initial beta")

    def test_successive_halving_matches_the_quadratic_example(self) -> None:
        function = parse_function("x**2 + 2*y**2 - 2*x - 4*y", "x, y")
        with patch.object(
            ParsedFunction, "gradient_at", autospec=True, side_effect=ParsedFunction.gradient_at
        ) as gradient_at:
            result = optimize(function, OptimizationMethod.GRADIENT_HALVING, (4, 3))

        self.assertTrue(result.converged)
        self.assertEqual(result.first_convergence_iteration, 2)
        self.assertFalse(result.completed_requested_iterations)
        self.assertEqual(result.message, "Stationary point reached")
        self.assertEqual([step.point for step in result.steps], [(4, 3), (1, -1), (1, 1)])
        self.assertEqual([step.value for step in result.steps], [14, 5, -3])
        self.assertEqual(result.steps[0].gradient, (6, 8))
        self.assertEqual([tuple(call.args[1]) for call in gradient_at.call_args_list],
                         [(4, 3), (1, -1), (1, 1)])
        self.assertEqual(
            [(t.step_size, t.point, t.value, t.accepted) for t in result.steps[1].trials],
            [(1, (-2, -5), 78, False), (0.5, (1, -1), 5, True)],
        )
        self.assertEqual([trial.step_size for trial in result.steps[2].trials], [0.5, 0.25])

    def test_successive_halving_continues_past_first_convergence(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (1,),
            OptimizationSettings(initial_step=0.25, tolerance=0.75, max_iterations=4),
        )

        self.assertEqual([step.point for step in result.steps], [(1,), (0.5,), (0.25,), (0.125,), (0.0625,)])
        self.assertEqual(result.first_convergence_iteration, 2)
        self.assertTrue(result.completed_requested_iterations)
        self.assertEqual(result.message, "Done")
        table = format_intermediate_details_html(result)
        self.assertEqual(table.count("Convergence reached"), 1)
        self.assertEqual(table.count("Accepted"), 4)

    def test_successive_halving_continues_from_a_nonstationary_start_within_tolerance(self) -> None:
        result = optimize(
            parse_function("x**2 / 8", "x"), OptimizationMethod.GRADIENT_HALVING, (1,),
        )
        self.assertEqual(len(result.steps) - 1, 10)
        self.assertTrue(result.completed_requested_iterations)

        result = optimize(
            parse_function("x**2", "x"), OptimizationMethod.GRADIENT_HALVING, (1,),
            OptimizationSettings(initial_step=0.25, tolerance=3, max_iterations=2),
        )
        self.assertEqual(result.first_convergence_iteration, 0)
        self.assertEqual(len(result.steps) - 1, 2)
        self.assertTrue(result.completed_requested_iterations)

    def test_successive_halving_reports_failed_trials_after_accepted_steps(self) -> None:
        result = optimize(
            parse_function("1 + x**2", "x"), OptimizationMethod.GRADIENT_HALVING, (2.0**-26,),
            OptimizationSettings(initial_step=0.25, max_iterations=3, max_halvings=2),
        )
        self.assertEqual(len(result.steps) - 1, 1)
        self.assertEqual(len(result.failed_trials), 3)
        self.assertFalse(result.completed_requested_iterations)
        self.assertIn("No improving step was found at iteration 2", result.message)
        table = format_intermediate_details_html(result)
        self.assertEqual(table.count("No improvement"), 3)
        self.assertEqual(table.count("Accepted"), 1)

    def test_successive_halving_keeps_failed_trials_out_of_the_trajectory(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (1,),
            OptimizationSettings(initial_step=2, halving_factor=0.75, max_halvings=2),
        )

        self.assertFalse(result.converged)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.final_point, (1,))
        self.assertEqual([trial.step_size for trial in result.failed_trials], [2, 1.5, 1.125])
        self.assertIn("limit of 2 halvings", result.message)
        self.assertIn("No improvement", format_intermediate_details_html(result))

    def test_successive_halving_stops_when_step_cannot_move_the_point(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (1,),
            OptimizationSettings(initial_step=1e-20),
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.final_point, (1,))
        self.assertEqual(len(result.failed_trials), 1)
        self.assertIn("too small", result.message)

    def test_successive_halving_reduces_beta_after_non_finite_function_value(self) -> None:
        function = parse_function("(sqrt(x) - 1)**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (4,),
            OptimizationSettings(initial_step=12, max_iterations=1),
        )

        self.assertEqual([trial.value for trial in result.steps[1].trials], [None, 0.0])
        self.assertEqual(result.final_point, (1,))

    def test_successive_halving_requires_a_strict_gradient_tolerance(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (1,),
            OptimizationSettings(tolerance=2, initial_step=0.25, max_iterations=1),
        )

        self.assertEqual(result.final_point, (0.5,))
        self.assertEqual(result.first_convergence_iteration, 1)

    def test_successive_halving_needs_no_trial_at_a_stationary_start(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (0,),
        )

        self.assertTrue(result.converged)
        self.assertIsNone(result.accepted_halving_step)
        self.assertEqual(result.failed_trials, ())
        self.assertEqual(result.first_convergence_iteration, 0)
        self.assertEqual(result.message, "Stationary point reached")

    def test_successive_halving_keeps_accepted_point_if_its_gradient_is_undefined(self) -> None:
        function = parse_function("sqrt(x)", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_HALVING, (1,),
            OptimizationSettings(initial_step=2),
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.final_point, (0,))
        self.assertTrue(result.accepted_halving_step.trials[0].accepted)
        self.assertFalse(result.completed_requested_iterations)
        self.assertIn("Stopped after iteration 1", result.message)

    def test_parabola_fitting_starts_at_beta_without_halving(self) -> None:
        function = parse_function("x1**2 + x2**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_PARABOLA,
            (1.0, 2.0),
            OptimizationSettings(initial_step=1.0, max_iterations=1),
        )

        phases = [trial.phase for trial in result.steps[1].trials]
        self.assertEqual(phases[:2], ["initial beta", "second sample (2β)"])
        self.assertNotIn("halving", phases)
        self.assertEqual(len(phases), 3)

    def test_parabola_fitting_solves_the_supplied_example(self) -> None:
        function = parse_function("x**2 + 2*y**2 - 2*x - 4*y", "x, y")
        result = optimize(function, OptimizationMethod.GRADIENT_PARABOLA, (4, 3))

        base, search = result.parabola_iterations[0]
        self.assertEqual(base.gradient, (6, 8))
        self.assertEqual([sample.point for sample in search.samples], [(4, 3), (2.5, 1), (1, -1)])
        self.assertEqual([sample.value for sample in search.samples], [14, -0.75, 5])
        self.assertAlmostEqual(search.fitted.step_size, 25 / 82)
        self.assertAlmostEqual(search.fitted.value, -51 / 41)
        self.assertIs(search.selected, search.fitted)
        np.testing.assert_allclose(result.steps[1].point, [89 / 41, 23 / 41])
        np.testing.assert_allclose(
            search.interpolation.evaluate(np.array([0.0, 0.25, 0.5])),
            [14, -0.75, 5],
        )
        self.assertTrue(all(step.samples[1].step_size == 0.25 for _, step in result.parabola_iterations))
        self.assertTrue(all(a.value > b.value for a, b in zip(result.steps, result.steps[1:])))
        self.assertTrue(result.converged)
        self.assertLess(result.steps[-1].gradient_norm, 1e-6)
        np.testing.assert_allclose(result.final_point, [1, 1], atol=1e-6)
        self.assertAlmostEqual(result.final_value, -3)

    def test_parabola_fitting_respects_iteration_limit(self) -> None:
        function = parse_function("x**2 + 2*y**2 - 2*x - 4*y", "x, y")
        result = optimize(
            function, OptimizationMethod.GRADIENT_PARABOLA, (4, 3),
            OptimizationSettings(initial_step=0.25, max_iterations=1),
        )

        self.assertFalse(result.converged)
        self.assertEqual(len(result.parabola_iterations), 1)
        self.assertIn("Maximum number of iterations", result.message)

    def test_parabola_fitting_uses_strict_gradient_tolerance(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_PARABOLA, (1,),
            OptimizationSettings(initial_step=0.25, tolerance=2),
        )

        self.assertTrue(result.converged)
        self.assertEqual(len(result.parabola_iterations), 1)
        self.assertEqual(result.final_point, (0.0,))

    def test_parabola_fitting_reports_invalid_function_values(self) -> None:
        function = parse_function("sqrt(x)", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_PARABOLA, (1,),
            OptimizationSettings(initial_step=3),
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.final_point, (1,))
        self.assertIsNotNone(result.failed_parabola)
        self.assertIn("invalid function value", result.message)
        self.assertIn("invalid function value", format_optimization_details_html(result, parabola_iteration=1))

    def test_html_details_show_exercise_style_beta_table(self) -> None:
        function = parse_function("(x1 - 2)**4 + (x1 - 2*x2)**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_HALVING,
            (0.0, 0.0),
            OptimizationSettings(max_iterations=1),
        )
        details = format_optimization_details_html(result)

        self.assertNotIn("Successive halving candidates", details)
        self.assertNotIn("Iteration results", details)

        line_search_details = format_intermediate_details_html(result)
        self.assertIn("Successive halving candidates", line_search_details)
        self.assertNotIn("Iteration results", line_search_details)
        self.assertIn("Current point", line_search_details)
        self.assertIn("Test point", line_search_details)
        self.assertIn("Decision", line_search_details)
        self.assertIn("Accepted", line_search_details)

    def test_newton_and_broyden_iteration_tables_are_intermediate_results(self) -> None:
        function = parse_function("x1**2 + x2**2", "x1, x2")
        for method in (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN):
            with self.subTest(method=method):
                result = optimize(function, method, (3.0, 1.0), OptimizationSettings(max_iterations=2))
                formulas = format_optimization_details_html(result)
                intermediate = format_intermediate_details_html(result)

                self.assertNotIn("Iteration results", formulas)
                self.assertIn("Iteration results", intermediate)
                self.assertIn("Step details", intermediate)

    def test_no_improving_step_stops_without_convergence(self) -> None:
        function = parse_function("x**2", "x")
        result = optimize(
            function, OptimizationMethod.GRADIENT_PARABOLA, (1.0,),
            OptimizationSettings(initial_step=1e-18),
        )
        self.assertFalse(result.converged)
        self.assertEqual(len(result.steps), 1)
        self.assertIsNotNone(result.failed_parabola)
        self.assertIn("No improving step", result.message)
