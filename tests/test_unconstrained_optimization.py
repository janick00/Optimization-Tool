"""Tests for unconstrained continuous optimization methods."""

import unittest

from topics.unconstrained_optimization.solver import (
    OptimizationMethod,
    OptimizationSettings,
    FUNCTION_LIBRARY,
    format_line_search_details_html,
    format_optimization_details_html,
    optimize,
    parse_function,
)


class UnconstrainedOptimizationTests(unittest.TestCase):
    def test_function_presets_include_lecture_and_exercise_examples(self) -> None:
        identifiers = {definition.identifier for definition in FUNCTION_LIBRARY}
        self.assertIn("bazaraa_shetty", identifiers)
        self.assertIn("exercise_9_function", identifiers)

    def test_parser_creates_gradient_and_hessian(self) -> None:
        function = parse_function("x1**2 + 2*x1*x2 + x2**2", "x1, x2")
        self.assertEqual(function.dimension, 2)
        self.assertEqual(tuple(map(str, function.gradient_expression)), ("2*x1 + 2*x2", "2*x1 + 2*x2"))
        self.assertEqual(function.gradient_at((1.0, 2.0)).tolist(), [6.0, 6.0])
        self.assertEqual(function.hessian_at((1.0, 2.0)).tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_all_methods_reach_quadratic_bowl_minimum(self) -> None:
        function = parse_function("x1**2 + x2**2", "x1, x2")
        settings = OptimizationSettings(tolerance=1e-8, max_iterations=20)
        for method in OptimizationMethod:
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
        self.assertTrue(any(step.trials for step in result.steps[1:]))

    def test_successive_halving_starts_at_beta_and_only_halves_or_doubles(self) -> None:
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
            [1.0, 0.5, 0.25, 0.125, 0.0625, 0.125],
        )
        self.assertEqual(
            [trial.phase for trial in first_iteration.trials],
            ["initial beta", "halving", "halving", "halving", "halving", "doubling"],
        )
        self.assertIn("β₀=1", first_iteration.note)

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

    def test_parabola_fitting_starts_at_beta_without_halving(self) -> None:
        function = parse_function("(x1 - 2)**4 + (x1 - 2*x2)**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_PARABOLA,
            (0.0, 0.0),
            OptimizationSettings(initial_step=1.0, max_iterations=1),
        )

        phases = [trial.phase for trial in result.steps[1].trials]
        self.assertEqual(phases[:2], ["initial beta", "doubling sample"])
        self.assertNotIn("halving", phases)
        self.assertIn("no halving", result.steps[1].note)

    def test_html_details_show_exercise_style_beta_table(self) -> None:
        function = parse_function("(x1 - 2)**4 + (x1 - 2*x2)**2", "x1, x2")
        result = optimize(
            function,
            OptimizationMethod.GRADIENT_HALVING,
            (0.0, 0.0),
            OptimizationSettings(max_iterations=1),
        )
        details = format_optimization_details_html(result)

        self.assertIn("Successive halving candidates", details)
        self.assertIn("Candidate", details)
        self.assertIn("done!", details)
        self.assertIn("β", details)

        line_search_details = format_line_search_details_html(result)
        self.assertIn("Successive halving candidates", line_search_details)
        self.assertIn("done!", line_search_details)
