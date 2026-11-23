from fractions import Fraction
from itertools import product
from random import Random
from unittest import TestCase

from topics.integer_programming.knapsack import (
    LECTURE, Item, Knapsack, SearchOrder, density_order, greedy_relaxation, knapsack_branch_and_bound, packing,
)
from topics.integer_programming.models import NodeStatus


class KnapsackTests(TestCase):
    def test_lecture_greedy_in_sorted_and_input_order(self):
        order = density_order(LECTURE)
        self.assertEqual(order, (0, 1, 2, 4, 3))
        self.assertEqual([LECTURE.items[i].weight for i in order], [5, 6, 12, 12, 10])
        self.assertEqual([LECTURE.items[i].value for i in order], [10, 9, 12, 9, 5])
        relaxed = greedy_relaxation(LECTURE)
        self.assertEqual(relaxed.lp.x, (1, 1, 1, 0, Fraction(1, 6)))
        self.assertEqual(tuple(relaxed.lp.x[i] for i in order), (1, 1, 1, Fraction(1, 6), 0))
        self.assertEqual((relaxed.lp.weight, relaxed.lp.value), (25, Fraction(65, 2)))
        self.assertEqual(relaxed.heuristic.x, (1, 1, 1, 0, 0))
        self.assertEqual((relaxed.heuristic.weight, relaxed.heuristic.value), (23, 31))

    def test_depth_first_replays_the_lecture_nodes(self):
        run = knapsack_branch_and_bound(LECTURE)
        self.assertIsNone(run.events[0].upper)
        processed = [event.changes[0] for event in run.events if event.phase == "relaxation"]
        self.assertEqual([node.fixed for node in processed], [
            (), ((4, 1),), ((4, 0),), ((4, 0), (3, 1)), ((4, 0), (3, 0)),
        ])
        # Slide 16 prints 27 for visit 4. Its displayed vector gives 10 + 9 + 12/3 + 5 = 28.
        self.assertEqual([node.relaxation.lp.value for node in processed], [Fraction(65, 2), 30, 32, 28, 31])
        self.assertEqual([node.status for node in run.nodes_at(len(run.events) - 1).values()],
                         [NodeStatus.FRACTIONAL, NodeStatus.PRUNED, NodeStatus.FRACTIONAL,
                          NodeStatus.PRUNED, NodeStatus.INTEGER])
        final = run.events[-1]
        self.assertEqual((final.outcome, final.upper, final.incumbent.value), ("optimal", 31, 31))

    def test_best_bound_prefers_the_highest_relaxed_child_bound(self):
        run = knapsack_branch_and_bound(LECTURE, SearchOrder.BEST_BOUND)
        visited = [event.changes[0] for event in run.events if event.phase == "relaxation"]
        self.assertEqual(visited[1].fixed, ((4, 0),))
        self.assertEqual(run.events[-1].incumbent.value, 31)

    def test_infeasible_fixings_and_integral_optimality_are_distinct(self):
        problem = Knapsack((Item("A", 5, 10), Item("B", 2, 1)), 3)
        run = knapsack_branch_and_bound(problem)
        pruned = [event.changes[0] for event in run.events if event.phase == "pruned"]
        self.assertEqual([node.status for node in pruned], [NodeStatus.INFEASIBLE, NodeStatus.INTEGER])
        self.assertEqual(pruned[0].relaxation.fixed_weight, 5)
        self.assertIsNone(pruned[0].relaxation.lp)
        self.assertEqual(run.events[-1].incumbent.x, (0, 1))

    def test_node_limit_and_objective_tolerance_keep_a_certified_gap(self):
        limited = knapsack_branch_and_bound(LECTURE, node_limit=1).events[-1]
        self.assertEqual(limited.outcome, "limit")
        self.assertGreater(limited.upper, limited.incumbent.value)
        approximate = knapsack_branch_and_bound(LECTURE, SearchOrder.BEST_BOUND, tolerance=Fraction(1)).events[-1]
        self.assertEqual(approximate.outcome, "tolerance")
        self.assertEqual(approximate.upper - approximate.incumbent.value, 1)

    def test_exact_results_and_bounds_against_binary_enumeration(self):
        random = Random(23)
        for _ in range(16):
            problem = Knapsack(tuple(Item(str(i), random.randint(1, 8), random.randint(-2, 15))
                                     for i in range(5)), random.randint(0, 17))
            feasible = [p for x in product((0, 1), repeat=5)
                        if (p := packing(problem, x)).weight <= problem.capacity]
            expected = max(point.value for point in feasible)
            for strategy in SearchOrder:
                run = knapsack_branch_and_bound(problem, strategy)
                self.assertEqual(run.events[-1].outcome, "optimal")
                self.assertEqual(run.events[-1].incumbent.value, expected)
                for event in run.events:
                    if event.upper is not None:
                        self.assertGreaterEqual(event.upper, expected)
                    if event.phase == "relaxation" and event.changes[0].relaxation.lp is not None:
                        node = event.changes[0]
                        candidate = node.relaxation.heuristic
                        self.assertLessEqual(candidate.weight, problem.capacity)
                        self.assertTrue(all(candidate.x[i] == val for i, val in node.fixed))
                        legal = [p.value for p in feasible if all(p.x[i] == val for i, val in node.fixed)]
                        self.assertGreaterEqual(node.relaxation.lp.value, max(legal))

    def test_validation_and_nonpositive_values(self):
        with self.assertRaisesRegex(ValueError, "positive weight"):
            Item("A", 0, 2)
        with self.assertRaisesRegex(ValueError, "nonnegative capacity"):
            Knapsack(LECTURE.items, -1)
        with self.assertRaisesRegex(ValueError, "distinct item"):
            greedy_relaxation(LECTURE, ((0, 1), (0, 0)))
        problem = Knapsack((Item("zero", 1, 0), Item("negative", 1, -3)), 5)
        self.assertEqual(greedy_relaxation(problem).lp.x, (0, 0))
