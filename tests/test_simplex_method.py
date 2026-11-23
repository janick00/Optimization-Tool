"""Exact start validation and pivot tests for the Simplex page."""

from fractions import Fraction
import unittest

from topics.linear_programming.problem_input import build_linear_program
from topics.simplex_method.formatting import format_calculations
from topics.simplex_method.solver import objective_value, solve_simplex


def make_problem(
    objective="3, 2", matrix="1, 2\n2, 1", rhs="8, 8", relations="<=, <=",
    domains=">=0, >=0", sense="max",
):
    return build_linear_program("general", sense, objective, matrix, rhs, relations, domains)


class SimplexStartTests(unittest.TestCase):
    def test_automatic_and_manual_find_the_same_optimum_from_different_vertices(self):
        problem = make_problem()
        automatic = solve_simplex(problem)
        manual = solve_simplex(problem, (0, 4))

        self.assertEqual(automatic.start.point, (0, 0))
        self.assertEqual(manual.start.point, (0, 4))
        self.assertEqual(manual.steps[0].point, (0, 4))
        self.assertEqual(manual.start.matrix, ((1, 2), (-1, 0)))
        self.assertEqual(manual.start.objective_value, 8)
        for result in (automatic, manual):
            self.assertEqual(result.status, "optimal")
            self.assertEqual(result.steps[-1].point, (Fraction(8, 3), Fraction(8, 3)))
            self.assertEqual(objective_value(problem, result.steps[-1].point), Fraction(40, 3))
        html = format_calculations(problem, manual)
        for item in ("Starting point:", "Feasibility check", "Active constraints:",
                     "Start basis B:", "Basis matrix", "Linear independence", "Start objective value:"):
            self.assertIn(item, html)

    def test_rejected_points_have_a_specific_reason_and_never_pivot(self):
        problem = make_problem()
        for point, reason in (
            ((10, 0), "Violated constraints"),
            ((-1, 0), "Violated variable domains"),
            ((1, 1), "Active rank 0; required rank 2"),
            ((2, 3), "Active rank 1; required rank 2"),
        ):
            with self.subTest(point=point):
                result = solve_simplex(problem, point)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.start.point, point)
                self.assertEqual(result.start.basis, ())
                self.assertEqual(result.steps, ())
                self.assertIn(reason, result.message)

    def test_degenerate_start_chooses_first_independent_rows_in_input_order(self):
        problem = make_problem(matrix="1, 1\n2, 2", rhs="0, 0")
        result = solve_simplex(problem, (0, 0))

        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.start.basis, (0, 2))
        self.assertEqual(result.start.active_rank, 2)
        self.assertEqual(len([check for check in result.start.checks if check.active]), 4)
        self.assertEqual(result.steps[-1].point, (0, 0))
        self.assertIn("first independent active rows", format_calculations(problem, result))

    def test_phase_one_handles_shifted_equalities_and_nonpositive_free_domains(self):
        problem = make_problem(
            objective="1, 0", matrix="1, 1\n0, 1", rhs="1, 2",
            relations="=, <=", domains="<=0, free", sense="min",
        )
        automatic = solve_simplex(problem)
        manual = solve_simplex(problem, (0, 1))
        for result in (automatic, manual):
            self.assertEqual(result.status, "optimal")
            self.assertEqual(result.steps[-1].point, (-1, 2))
            self.assertTrue(all(check.satisfied for check in result.start.checks))
        self.assertEqual(manual.steps[0].point, (0, 1))
        self.assertEqual(solve_simplex(problem, ("1/2", "1/2")).status, "error")

    def test_phase_one_finds_vertex_with_exact_rational_coordinates(self):
        problem = make_problem(
            objective="1", matrix="3\n1", rhs="1, 2",
            relations=">=, <=", domains="free", sense="min",
        )
        result = solve_simplex(problem)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.steps[-1].point, (Fraction(1, 3),))
        manual = solve_simplex(problem, ("1/3",))
        self.assertEqual(manual.start.point, (Fraction(1, 3),))
        self.assertEqual(manual.steps[0].point, manual.start.point)

    def test_automatic_reports_infeasible_unbounded_and_no_vertex(self):
        infeasible = make_problem(
            "1", "1\n1", "2, 1", ">=, <=", "free",
        )
        unbounded = make_problem("1", "1", "2", ">=", "free")
        no_vertex = make_problem(matrix="1, 0", rhs="2", relations="<=", domains="free, free")
        self.assertIn("no feasible point", solve_simplex(infeasible).message)
        self.assertEqual(solve_simplex(unbounded).status, "unbounded")
        self.assertIn("No full-rank vertex basis", solve_simplex(no_vertex).message)

    def test_degenerate_pivots_terminate_on_a_cycling_example(self):
        problem = make_problem(
            objective="10, -57, -9, -24",
            matrix="1/2, -11/2, -5/2, 9\n1/2, -3/2, -1/2, 1\n1, 0, 0, 0",
            rhs="0, 0, 1", relations="<=, <=, <=", domains=">=0, >=0, >=0, >=0",
        )
        result = solve_simplex(problem, (0, 0, 0, 0))
        self.assertEqual(result.status, "optimal")
        self.assertEqual(objective_value(problem, result.steps[-1].point), 1)
        self.assertTrue(any(step.distance == 0 for step in result.steps))
