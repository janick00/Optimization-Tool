"""Tests for linear-programming form conversion."""

import unittest
from fractions import Fraction

from topics.linear_programming.form_conversion import convert_problem
from topics.linear_programming.formatting import DisplayNotation, format_linear_program_html
from topics.linear_programming.models import (
    ConstraintRelation,
    ObjectiveSense,
    ProblemForm,
    VariableDomain,
)
from topics.linear_programming.problem_input import build_linear_program


class FormConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = build_linear_program(
            source_form=ProblemForm.GENERAL,
            objective_sense=ObjectiveSense.MAXIMIZE,
            objective_text="3, 2",
            matrix_text="1, 2\n2, 1",
            right_hand_side_text="8, 8",
            relation_text="<=, >=",
            domain_text=">=0, free",
            variable_names_text="x1, x2",
        )

    def test_general_input_is_parsed(self) -> None:
        self.assertEqual(self.problem.objective, (Fraction(3), Fraction(2)))
        self.assertEqual(self.problem.constraints[1].relation, ConstraintRelation.GREATER_EQUAL)
        self.assertEqual(self.problem.variable_domains[1], VariableDomain.FREE)

    def test_conversion_to_standard_form_adds_slack_and_surplus(self) -> None:
        result = convert_problem(self.problem, ProblemForm.STANDARD, ProblemForm.GENERAL)
        converted = result.problem
        self.assertEqual(
            converted.variable_names,
            ("x1", "x2_plus", "x2_minus", "s_1", "u_2"),
        )
        self.assertEqual(converted.variable_domains, (VariableDomain.NONNEGATIVE,) * 5)
        self.assertEqual(
            tuple(constraint.relation for constraint in converted.constraints),
            (ConstraintRelation.EQUAL, ConstraintRelation.EQUAL),
        )
        self.assertEqual(converted.constraints[0].coefficients[-2:], (Fraction(1), Fraction(0)))
        self.assertEqual(converted.constraints[1].coefficients[-1], Fraction(-1))

    def test_conversion_to_canonical_form_uses_maximum_less_equal_direction(self) -> None:
        result = convert_problem(self.problem, ProblemForm.CANONICAL, ProblemForm.GENERAL)
        converted = result.problem
        self.assertTrue(all(domain is VariableDomain.NONNEGATIVE for domain in converted.variable_domains))
        self.assertTrue(
            all(constraint.relation is ConstraintRelation.LESS_EQUAL for constraint in converted.constraints)
        )
        self.assertGreaterEqual(len(converted.constraints), 2)

    def test_conversion_to_inequality_form_moves_domain_constraints(self) -> None:
        result = convert_problem(self.problem, ProblemForm.INEQUALITY, ProblemForm.GENERAL)
        converted = result.problem
        self.assertEqual(converted.variable_domains, (VariableDomain.FREE, VariableDomain.FREE))
        self.assertTrue(
            all(constraint.relation is ConstraintRelation.LESS_EQUAL for constraint in converted.constraints)
        )
        self.assertEqual(len(converted.constraints), 3)

    def test_every_target_form_is_available(self) -> None:
        for target_form in ProblemForm:
            with self.subTest(target_form=target_form):
                result = convert_problem(self.problem, target_form, ProblemForm.GENERAL)
                self.assertEqual(result.target_form, target_form)
                self.assertGreater(len(result.problem.objective), 0)

    def test_html_renderer_supports_all_display_notations(self) -> None:
        for notation in DisplayNotation:
            with self.subTest(notation=notation):
                html_output = format_linear_program_html(
                    self.problem,
                    ProblemForm.GENERAL,
                    notation,
                )
                self.assertIn("<html>", html_output)
                self.assertIn("Subject to", html_output)
                label = ProblemForm.GENERAL.label if notation is DisplayNotation.EXPANDED else notation.label
                self.assertIn(label, html_output)
