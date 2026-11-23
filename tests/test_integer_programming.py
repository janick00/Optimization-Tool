"""Exact search, event chronology and large-tree layout."""

from dataclasses import replace
from fractions import Fraction
import unittest

from topics.integer_programming.feasibility import check_integer_point, relaxation_bounds
from topics.integer_programming.formatting import (
    format_node, format_relaxation_range, format_simplex_trace,
)
from topics.integer_programming.models import Bounds, Domain, EventKind, Node, NodeStatus
from topics.integer_programming.solver import branch_and_bound, relaxation
from topics.integer_programming.tree_view import BOX_WIDTH, tree_positions
from topics.linear_programming.problem_input import build_linear_program
from topics.simplex_method.solver import solve_simplex


def problem(objective="3,2", matrix="1,2\n2,1", rhs="8,8", relations="<=,<=", sense="max"):
    domains = ", ".join(">=0" for _ in objective.split(","))
    return build_linear_program("general", sense, objective, matrix, rhs, relations, domains)


def integer_domains(count=2):
    return (Domain.INTEGER,) * count


class BranchAndBoundTests(unittest.TestCase):
    def test_worked_example_uses_exact_lp_data_and_chronological_bounds(self):
        run = branch_and_bound(problem(), integer_domains())
        self.assertEqual(run.status, "optimal")
        self.assertEqual([event.kind for event in run.events], [
            EventKind.ROOT_CREATED, EventKind.LP_SOLVED, EventKind.INTEGRALITY_CHECKED,
            EventKind.BOUNDS_UPDATED, EventKind.BRANCH_SELECTED, EventKind.CHILDREN_CREATED,
            EventKind.LP_SOLVED, EventKind.INTEGRALITY_CHECKED, EventKind.BOUNDS_UPDATED,
            EventKind.LP_SOLVED, EventKind.INTEGRALITY_CHECKED, EventKind.BOUNDS_UPDATED,
            EventKind.FINISHED,
        ])
        initial = run.frame(0)
        self.assertEqual(list(initial.nodes), [1])
        self.assertEqual(initial.nodes[1].status, NodeStatus.OPEN)
        self.assertIsNone(initial.nodes[1].simplex)
        self.assertEqual(initial.event.bounds, Bounds())
        root = run.frame(1).nodes[1]
        self.assertEqual(root.solution, (Fraction(8, 3), Fraction(8, 3)))
        self.assertEqual(root.value, Fraction(40, 3))
        self.assertIsNone(root.branch_variable)
        self.assertIsNone(root.fractional)
        self.assertEqual(root.status, NodeStatus.LP_SOLVED)
        self.assertEqual(run.frame(2).nodes[1].fractional, (0, 1))
        self.assertIsNone(run.frame(2).nodes[1].branch_variable)
        self.assertIsNone(run.frame(2).event.bounds.upper)
        self.assertEqual(run.frame(3).event.bounds.upper, Fraction(40, 3))
        self.assertEqual(run.frame(4).nodes[1].branch_variable, 0)
        self.assertEqual(list(run.frame(4).nodes), [1])
        branched = run.frame(5)
        self.assertEqual(set(branched.nodes), {1, 2, 3})
        self.assertIsNone(branched.nodes[2].solution)
        self.assertIsNone(branched.event.incumbent)
        for identifier, condition in ((2, "x1 ≤ 2"), (3, "x1 ≥ 3")):
            node = branched.nodes[identifier]
            self.assertEqual(node.parent, 1)
            self.assertEqual(node.branches[-1].label(node.problem.variable_names), condition)
            self.assertEqual(node.bounds.upper, Fraction(40, 3))
        self.assertIsNone(run.frame(7).event.incumbent)
        self.assertEqual(run.frame(8).event.bounds, Bounds(Fraction(12), Fraction(40, 3)))
        self.assertEqual(run.frame(11).nodes[3].status, NodeStatus.INTEGER)
        self.assertIsNone(run.frame(11).event.outcome)
        final = run.frame(12)
        self.assertEqual(final.nodes[2].solution, (2, 3))
        self.assertEqual(final.nodes[2].value, 12)
        self.assertEqual(final.nodes[3].solution, (3, 2))
        self.assertEqual(final.nodes[3].value, 13)
        self.assertEqual(final.nodes[3].status, NodeStatus.OPTIMAL)
        self.assertEqual(final.event.bounds, Bounds(Fraction(13), Fraction(13)))
        self.assertEqual(final.event.incumbent.point, (3, 2))
        self.assertEqual(final.event.outcome, "optimal")
        self.assertEqual(len(final.nodes[3].simplex.phases), 2)
        trace = format_simplex_trace(final.nodes[3].simplex)
        for label in ("Phase I", "Phase II", "Blocking ratios", "Inverse basis matrix"):
            self.assertIn(label, trace)
        self.assertNotIn("40/3", format_node(initial.nodes[1], initial, integer_domains()))

    def test_multilevel_search_and_infeasible_children(self):
        lp = problem("1,1", "1,0\n0,1", "5/2,5/2")
        run = branch_and_bound(lp, integer_domains())
        final = run.frame(len(run.events) - 1)
        self.assertEqual(final.event.incumbent.point, (2, 2))
        self.assertTrue(any(len(node.branches) == 2 for node in final.nodes.values()))
        self.assertTrue(any(node.status is NodeStatus.INFEASIBLE for node in final.nodes.values()))

    def test_bound_pruning_and_minimization(self):
        run = branch_and_bound(problem("2,3"), integer_domains())
        final = run.frame(len(run.events) - 1)
        self.assertEqual(final.nodes[3].status, NodeStatus.PRUNED)
        self.assertIn("cannot improve", final.nodes[3].reason)
        prune_step = next(i for i, event in enumerate(run.events)
                          if event.kind is EventKind.PRUNED and event.focus == 3)
        before_prune = run.frame(prune_step - 1)
        self.assertEqual(before_prune.event.kind, EventKind.BOUNDS_UPDATED)
        self.assertEqual(before_prune.nodes[3].status, NodeStatus.INTEGER)
        self.assertNotIn("cannot improve", before_prune.nodes[3].reason)
        self.assertEqual(final.event.incumbent.point, (2, 3))
        lp = problem("1,1", "1,0\n0,1", "1/2,1/2", ">=,>=", "min")
        run = branch_and_bound(lp, integer_domains())
        self.assertEqual(run.frame(3).event.bounds.lower, 1)
        self.assertIsNone(run.frame(3).event.bounds.upper)
        final = run.frame(len(run.events) - 1)
        self.assertEqual(final.event.incumbent.point, (1, 1))
        self.assertEqual(final.event.bounds, Bounds(Fraction(2), Fraction(2)))

    def test_mixed_binary_and_fraction_below_float_resolution(self):
        run = branch_and_bound(problem(), (Domain.CONTINUOUS, Domain.INTEGER))
        self.assertEqual(run.frame(4).nodes[1].branch_variable, 1)
        binary = problem("1", "1", "3/2", "<=")
        self.assertEqual(len(relaxation(binary, (Domain.BINARY,)).constraints), 3)
        run = branch_and_bound(binary, (Domain.BINARY,))
        self.assertEqual(run.frame(len(run.events) - 1).event.incumbent.point, (1,))
        near = problem("1", "1", "10000000000000001/10000000000000000", "<=")
        run = branch_and_bound(near, integer_domains(1))
        self.assertEqual(run.frame(2).nodes[1].status, NodeStatus.FRACTIONAL)
        self.assertEqual(run.frame(len(run.events) - 1).event.incumbent.point, (1,))

    def test_limit_and_unbounded_relaxation_never_claim_optimality(self):
        limited = branch_and_bound(problem(), integer_domains(), node_limit=2)
        final = limited.frame(len(limited.events) - 1)
        self.assertEqual(limited.status, "limit")
        self.assertEqual(final.nodes[3].status, NodeStatus.OPEN)
        self.assertEqual(final.event.bounds, Bounds(Fraction(12), Fraction(40, 3)))
        self.assertEqual(final.event.outcome, "limit")
        self.assertNotIn(NodeStatus.OPTIMAL, [node.status for node in final.nodes.values()])
        unbounded = branch_and_bound(problem("1", "1", "0", ">="), integer_domains(1))
        self.assertEqual(unbounded.status, "unresolved")
        self.assertEqual(unbounded.frame(1).nodes[1].status, NodeStatus.UNRESOLVED)

    def test_infeasible_integer_problem_and_root_relaxation(self):
        integer_infeasible = problem("1", "1", "1/2", "=")
        run = branch_and_bound(integer_infeasible, integer_domains(1))
        self.assertEqual(run.status, "infeasible")
        self.assertTrue(all(node.status is NodeStatus.INFEASIBLE
                            for node in list(run.frame(len(run.events) - 1).nodes.values())[1:]))
        lp_infeasible = problem("1", "1", "-1", "<=")
        run = branch_and_bound(lp_infeasible, integer_domains(1))
        self.assertEqual(run.frame(1).nodes[1].status, NodeStatus.LP_SOLVED)
        self.assertEqual(run.frame(2).event.kind, EventKind.PRUNED)
        self.assertEqual(run.frame(2).nodes[1].status, NodeStatus.INFEASIBLE)

    def test_manual_incumbent_is_validated_and_independent_of_root(self):
        lp = problem()
        run = branch_and_bound(lp, integer_domains(), initial_incumbent=(3, 2))
        self.assertIsNone(run.frame(0).event.incumbent)
        seeded = run.frame(1)
        self.assertEqual(seeded.event.kind, EventKind.INITIAL_INCUMBENT)
        self.assertEqual(seeded.event.bounds, Bounds(lower=Fraction(13)))
        self.assertEqual(seeded.event.incumbent.point, (3, 2))
        self.assertIsNone(seeded.event.incumbent.node)
        self.assertIsNone(seeded.nodes[1].solution)
        self.assertEqual(run.frame(2).nodes[1].solution, (Fraction(8, 3), Fraction(8, 3)))
        final = run.frame(len(run.events) - 1)
        self.assertEqual(final.event.outcome, "optimal")
        self.assertEqual(final.event.incumbent.point, (3, 2))
        self.assertEqual(final.nodes[3].status, NodeStatus.PRUNED)
        self.assertIn("Global optimum", format_node(final.nodes[1], final, integer_domains()))

        for point, message in (((3,), "exactly 2"), ((Fraction(3, 2), 2), "must be an integer"),
                               ((4, 3), "infeasible"), ((-1, 0), "infeasible")):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, message):
                branch_and_bound(lp, integer_domains(), initial_incumbent=point)
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            branch_and_bound(lp, (Domain.BINARY, Domain.CONTINUOUS), initial_incumbent=(2, 0))

        upgraded = branch_and_bound(lp, integer_domains(), initial_incumbent=(0, 0))
        self.assertEqual(upgraded.events[-1].incumbent.node, 3)
        mixed = branch_and_bound(lp, (Domain.INTEGER, Domain.CONTINUOUS), initial_incumbent=(1, "3/2"))
        self.assertEqual(mixed.frame(1).event.incumbent.point, (1, Fraction(3, 2)))

    def test_large_tree_boxes_do_not_overlap(self):
        root = Node(1, None, (), problem(), 0)
        nodes = [root] + [replace(root, id=i, parent=i // 2) for i in range(2, 512)]
        positions = tree_positions(nodes)
        levels = {}
        for x, y in positions.values():
            levels.setdefault(y, []).append(x)
        self.assertEqual(len(positions), 511)
        for level in levels.values():
            level.sort()
            self.assertTrue(all(b - a > BOX_WIDTH for a, b in zip(level, level[1:])))


class RelaxationComparisonTests(unittest.TestCase):
    def test_maximization_uses_lp_upper_and_manual_lower(self):
        lp = problem()
        domains = integer_domains()
        relaxed = relaxation(lp, domains)
        result = solve_simplex(relaxed)
        point = check_integer_point(lp, domains, (Fraction(3), Fraction(2)))
        self.assertTrue(point.feasible)
        self.assertEqual(point.value, 13)
        self.assertEqual(relaxation_bounds(relaxed, result, point), Bounds(Fraction(13), Fraction(40, 3)))
        self.assertIn("13 ≤ z<sup>*</sup><sub>ILP</sub> ≤ 40/3",
                      format_relaxation_range(relaxed, result, point))

    def test_minimization_uses_lp_lower_and_manual_upper(self):
        lp = problem("3,2", "1,1", "1/2", ">=", "min")
        domains = integer_domains()
        relaxed = relaxation(lp, domains)
        result = solve_simplex(relaxed)
        point = check_integer_point(lp, domains, (Fraction(0), Fraction(1)))
        self.assertEqual(result.steps[-1].point, (Fraction(0), Fraction(1, 2)))
        self.assertEqual(relaxation_bounds(relaxed, result, point), Bounds(Fraction(1), Fraction(2)))
        self.assertIn("1 ≤ z<sup>*</sup><sub>ILP</sub> ≤ 2",
                      format_relaxation_range(relaxed, result, point))

    def test_invalid_point_lists_all_violated_rows_and_has_no_bound(self):
        lp = problem()
        domains = integer_domains()
        relaxed = relaxation(lp, domains)
        result = solve_simplex(relaxed)
        point = check_integer_point(lp, domains, (Fraction(9, 2), Fraction(3)))
        self.assertFalse(point.feasible)
        self.assertIn("x1", point.integrality_errors[0])
        self.assertEqual(point.constraint_errors, ("C1: 21/2 ≤ 8 is violated.", "C2: 12 ≤ 8 is violated."))
        self.assertEqual(relaxation_bounds(relaxed, result, point), Bounds(upper=Fraction(40, 3)))

    def test_binary_and_sign_bounds_with_a_continuous_coordinate(self):
        lp = problem("1,1", "1,1", "3/2", "=")
        domains = (Domain.BINARY, Domain.CONTINUOUS)
        good = check_integer_point(lp, domains, (Fraction(1), Fraction(1, 2)))
        fractional_binary = check_integer_point(lp, domains, (Fraction(1, 2), Fraction(1)))
        outside = check_integer_point(lp, domains, (Fraction(2), Fraction(-1, 2)))
        self.assertEqual(good.value, Fraction(3, 2))
        self.assertIn("Binary", fractional_binary.integrality_errors[0])
        self.assertEqual(len(outside.domain_errors), 2)
        self.assertEqual(outside.constraint_errors, ())
        self.assertFalse(outside.feasible)

    def test_no_finite_lp_bound_when_relaxation_is_unbounded(self):
        lp = problem("1", "1", "0", ">=")
        domains = integer_domains(1)
        result = solve_simplex(lp)
        point = check_integer_point(lp, domains, (Fraction(3),))
        self.assertEqual(result.status, "unbounded")
        self.assertEqual(relaxation_bounds(lp, result, point), Bounds(lower=Fraction(3)))
